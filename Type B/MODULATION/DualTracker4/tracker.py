import cv2
import numpy as np
from typing import Tuple, Optional, List


class DualKCFByteTrack:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.reset_state()

    def reset_state(self):
        self.kcf_tracker = None
        self.bytetrack_active = False
        self.tracking_active = False
        self.roi1 = None
        self.lost_frames = 0
        self.best_roi2_bbox = None  # 🔥 ROI2 저장
        self.fixed_roi_size = None
        self.bbox_history: List[Tuple[int, int]] = []
        self.roi_adjust_frame_count = 0

    def init_from_click(self, frame, x: int, y: int) -> bool:
        half = 100
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        w = min(frame.shape[1] - x1, 2 * half)
        h = min(frame.shape[0] - y1, 2 * half)

        init_bbox = (x1, y1, w, h)

        self.kcf_tracker = cv2.legacy.TrackerKCF_create()
        success = self.kcf_tracker.init(frame, init_bbox)

        self.bytetrack_active = True
        self.tracking_active = True
        self.lost_frames = 0
        self.roi1 = init_bbox
        self.bbox_history = [(w, h)]
        self.fixed_roi_size = (
            int(w * self.config.ROI_SCALE),
            int(h * self.config.ROI_SCALE),
        )
        self.roi_adjust_frame_count = 0

        print(f"Dual Tracker 초기화: KCF {init_bbox}, ByteTrack ON")
        return success

    def update(self, frame) -> Tuple[bool, Optional[Tuple]]:
        if not self.tracking_active or self.kcf_tracker is None:
            return False, None

        success, bbox = self.kcf_tracker.update(frame)
        if not success:
            self.lost_frames += 1
            return False, None

        x1, y1, w, h = map(int, bbox)
        self.roi1 = (x1, y1, w, h)

        # BBox history 갱신
        self.bbox_history.append((w, h))
        if len(self.bbox_history) > self.config.BBOX_HISTORY_LEN:
            self.bbox_history.pop(0)

        avg_w = int(np.mean([wh[0] for wh in self.bbox_history]))
        avg_h = int(np.mean([wh[1] for wh in self.bbox_history]))
        self.fixed_roi_size = (
            int(avg_w * self.config.ROI_SCALE),
            int(avg_h * self.config.ROI_SCALE),
        )

        self.best_roi2_bbox = None
        target_found = False

        # ByteTrack (매 프레임, conf=0.3)
        if self.bytetrack_active:
            roi_crop = frame[y1:y1 + h, x1:x1 + w]
            if roi_crop.size > 0:
                bt_results = self.model.track(
                    roi_crop,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=self.config.CONF_THRESHOLD,
                    verbose=False,
                )[0]

                if bt_results.boxes is not None:
                    boxes = bt_results.boxes.xyxy.cpu().numpy()   # (N, 4)
                    confs = bt_results.boxes.conf.cpu().numpy()   # (N,)
                    clss = bt_results.boxes.cls.cpu().numpy()     # (N,)

                    # ROI2 중심 (crop 좌표계)
                    cx_crop, cy_crop = w / 2.0, h / 2.0

                    best_score = float("inf")
                    best_box = None

                    # 클래스 우선순위 dict: {class_id: priority_index}
                    priority_map = {
                        cls_id: idx
                        for idx, cls_id in enumerate(self.config.CLASS_PRIORITY)
                    }

                    for i, box in enumerate(boxes):
                        conf_i = float(confs[i])
                        cls_i = int(clss[i])

                        if conf_i < self.config.CONF_THRESHOLD:
                            continue

                        x1b, y1b, x2b, y2b = box
                        cx = (x1b + x2b) / 2.0
                        cy = (y1b + y2b) / 2.0
                        dist = ((cx - cx_crop) ** 2 + (cy - cy_crop) ** 2) ** 0.5

                        # DIST_THRESHOLD는 필요하면 필터로 계속 사용
                        if dist > self.config.DIST_THRESHOLD:
                            continue

                        # 클래스 우선순위 인덱스 (없으면 제일 뒤로 보냄)
                        prio_idx = priority_map.get(
                            cls_i, len(priority_map)
                        )

                        # 최종 스코어: 클래스 우선, 같은 클래스에서는 중심과의 거리
                        score = prio_idx * 10000.0 + dist

                        if score < best_score:
                            best_score = score
                            best_box = box

                    if best_box is not None:
                        bx1 = int(best_box[0])
                        by1 = int(best_box[1])
                        bx2 = int(best_box[2])
                        by2 = int(best_box[3])
                        bw = bx2 - bx1
                        bh = by2 - by1

                        self.best_roi2_bbox = (bx1, by1, bw, bh)
                        target_found = True
                        self.lost_frames = 0
                    else:
                        self.best_roi2_bbox = None

        # 🔥 ROI 조정: YOLO 없이 best_roi2_bbox 재사용 (2ms CPU)
        self.roi_adjust_frame_count += 1
        if (
            self.roi_adjust_frame_count >= self.config.roi_adjust_interval
            and self.best_roi2_bbox is not None
        ):
            self.roi_adjust_frame_count = 0
            bx1, by1, bw, bh = self.best_roi2_bbox

            new_w = int(bw * self.config.ROI_SCALE)
            new_h = int(bh * self.config.ROI_SCALE)
            new_cx = x1 + bx1 + bw // 2
            new_cy = y1 + by1 + bh // 2

            new_x1 = max(0, new_cx - new_w // 2)
            new_y1 = max(0, new_cy - new_h // 2)
            new_x2 = min(frame.shape[1], new_x1 + new_w)
            new_y2 = min(frame.shape[0], new_y1 + new_h)

            new_roi1 = (new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1)

            self.kcf_tracker = cv2.legacy.TrackerKCF_create()
            kcf_success = self.kcf_tracker.init(frame, new_roi1)
            self.roi1 = new_roi1

            print(
                f"🚀 ROI1 Adjust (YOLO-FREE): {new_roi1} "
                f"(bw:{bw}x{bh}) {'OK' if kcf_success else 'FAIL'}"
            )

        return target_found, self.roi1

    def toggle_bytetrack(self):
        self.bytetrack_active = not self.bytetrack_active
        print(f"ByteTrack {'ON' if self.bytetrack_active else 'OFF'}")

    def is_tracking_valid(self) -> bool:
        return self.tracking_active and self.roi1 is not None



