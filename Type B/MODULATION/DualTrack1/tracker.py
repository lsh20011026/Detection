# tracker.py
import cv2
import numpy as np
from typing import Tuple, Optional

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
        self.fixed_roi_size = None
        self.center_history = []
        self.bbox_history = []
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
        self.center_history = [((x1 + w//2), (y1 + h//2))]
        self.fixed_roi_size = (int(w * self.config.ROI_SCALE), int(h * self.config.ROI_SCALE))
        self.roi_adjust_frame_count = 0

        print(f"Dual Tracker 초기화: KCF {init_bbox}, ByteTrack ON")
        return success

    def update(self, frame, mouse_callback_ctx=None) -> Tuple[bool, Optional[Tuple]]:
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
        self.fixed_roi_size = (int(avg_w * self.config.ROI_SCALE), int(avg_h * self.config.ROI_SCALE))

        target_found = False
        roi_crop = frame[y1:y1+h, x1:x1+w]
        if roi_crop.size > 0 and self.bytetrack_active:
            bt_results = self.model.track(
                roi_crop,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.config.CONF_THRESHOLD
            )[0]

            if bt_results.boxes is not None:
                boxes = bt_results.boxes.xyxy.cpu().numpy()
                confs = bt_results.boxes.conf.cpu().numpy()
                ids = bt_results.boxes.id.cpu().numpy() if bt_results.boxes.id is not None else None

                min_dist = float('inf')
                best_box = None
                best_id = None

                cx_crop, cy_crop = w//2, h//2
                for i, box in enumerate(boxes):
                    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                    dist = ((cx-cx_crop)**2 + (cy-cy_crop)**2)**0.5
                    if confs[i] > self.config.CONF_THRESHOLD and dist < self.config.DIST_THRESHOLD:
                        score = dist * (1 - confs[i])
                        if score < min_dist:
                            min_dist = score
                            best_box = box
                            best_id = int(ids[i]) if ids is not None else -1

                if best_box is not None:
                    bx1, by1, bx2, by2 = map(int, best_box)
                    ox1, oy1, ox2, oy2 = x1+bx1, y1+by1, x1+bx2, y1+by2
                    target_found = True
                    self.lost_frames = 0

        # ROI 조정 주기
        self.roi_adjust_frame_count += 1
        if self.roi_adjust_frame_count >= self.config.roi_adjust_interval:
            self.roi_adjust_frame_count = 0
            if self.roi1 and self.bytetrack_active:
                x1, y1, w, h = self.roi1
                roi_crop = frame[y1:y1+h, x1:x1+w]
                if roi_crop.size > 0:
                    results = self.model.track(
                        roi_crop,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=self.config.HIGH_CONF_THRESHOLD
                    )[0]
                    if results.boxes is not None:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        confs = results.boxes.conf.cpu().numpy()
                        best_box = None
                        max_conf = 0
                        cx_crop, cy_crop = w//2, h//2
                        for i, box in enumerate(boxes):
                            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                            dist = ((cx-cx_crop)**2 + (cy-cy_crop)**2)**0.5
                            if confs[i] > max_conf and dist < self.config.DIST_THRESHOLD * 0.5:
                                max_conf = confs[i]
                                best_box = box

                        if best_box is not None:
                            bx1, by1, bx2, by2 = map(int, best_box)
                            obj_w, obj_h = bx2-bx1, by2-by1
                            obj_cx_crop = bx1 + obj_w//2
                            obj_cy_crop = by1 + obj_h//2
                            new_cx = x1 + obj_cx_crop
                            new_cy = y1 + obj_cy_crop
                            new_w = int(obj_w * self.config.ROI_SCALE)
                            new_h = int(obj_h * self.config.ROI_SCALE)
                            new_x1 = max(0, new_cx - new_w//2)
                            new_y1 = max(0, new_cy - new_h//2)
                            new_x2 = min(frame.shape[1], new_x1 + new_w)
                            new_y2 = min(frame.shape[0], new_y1 + new_h)
                            new_roi1 = (new_x1, new_y1, new_x2-new_x1, new_y2-new_y1)

                            self.kcf_tracker = cv2.legacy.TrackerKCF_create()
                            kcf_success = self.kcf_tracker.init(frame, new_roi1)
                            self.roi1 = new_roi1
                            print(
                                f"ROI1 Adjust: {new_roi1} (conf:{max_conf:.2f}, "
                                f"obj_c:{int(new_cx)},{int(new_cy)}) {'OK' if kcf_success else 'FAIL'}"
                            )

        return target_found, self.roi1

    def toggle_bytetrack(self):
        self.bytetrack_active = not self.bytetrack_active
        print(f"ByteTrack {'ON' if self.bytetrack_active else 'OFF'}")

    def is_tracking_valid(self) -> bool:
        return self.tracking_active and self.roi1 is not None


