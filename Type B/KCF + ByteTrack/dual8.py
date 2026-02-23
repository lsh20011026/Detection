import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass

@dataclass
class CameraInfo:
    index: int
    width: int
    height: int

class CameraManager:
    def __init__(self, cam_width=640, cam_height=480):
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cameras = []
        self.current_camera = None
        self.cap = None
        self.frame_w = 0
        self.frame_h = 0

    def detect_available_cameras(self):
        self.cameras = []
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    self.cameras.append(CameraInfo(i, w, h))
                cap.release()
        print(f"사용 가능 카메라: {[c.index for c in self.cameras]}")

    def init_camera(self, cam_index):
        if self.cap:
            self.cap.release()

        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"카메라 열기 실패: index={cam_index}")
            self.cap = None
            return

        self.cap = cap
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_camera = next(
            (c for c in self.cameras if c.index == cam_index),
            CameraInfo(cam_index, self.frame_w, self.frame_h)
        )
        print(f"카메라 오픈: index={cam_index} ({self.frame_w}x{self.frame_h})")

    def switch_to_next(self):
        if len(self.cameras) <= 1:
            print("전환할 카메라 없음")
            return

        if self.current_camera is None:
            print("current_camera 없음 → 첫 카메라부터 시작")
            self.init_camera(self.cameras[0].index)
            return

        cur_idx = self.cameras.index(self.current_camera)
        next_idx = (cur_idx + 1) % len(self.cameras)
        next_cam = self.cameras[next_idx]
        print(f"카메라 전환 요청 → index={next_cam.index}")
        self.init_camera(next_cam.index)

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def cleanup(self):
        if self.cap:
            self.cap.release()
            self.cap = None

model = YOLO("/home/nes/yolo11n.engine", task="detect")

camera_manager = CameraManager(cam_width=640, cam_height=480)
camera_manager.detect_available_cameras()

if not camera_manager.cameras:
    print("사용 가능한 카메라가 없습니다. 종료합니다.")
    exit(0)

camera_manager.init_camera(camera_manager.cameras[0].index)
cap = camera_manager.cap

frame_buffer = None
kcf_tracker = None
bytetrack_active = False
tracking_active = False
roi1 = None
lost_frames = 0
KEEP_FRAMES = 30

roi_adjust_interval = 20
roi_adjust_frame_count = 0
ROI_SCALE = 1.5

roi_scale = 1.5
fixed_roi_size = None
center_history = []
bbox_history = []
BBOX_HISTORY_LEN = 5
DIST_THRESHOLD = 100
CONF_THRESHOLD = 0.3
HIGH_CONF_THRESHOLD = 0.7

fps_start = cv2.getTickCount()

def reset_tracking_state():
    global kcf_tracker, bytetrack_active, tracking_active, roi1
    global lost_frames, fixed_roi_size, center_history, bbox_history, roi_adjust_frame_count

    tracking_active = False
    kcf_tracker = None
    bytetrack_active = False
    roi1 = None
    fixed_roi_size = None
    center_history = []
    bbox_history = []
    lost_frames = 0
    roi_adjust_frame_count = 0
    print("추적 상태 리셋")

def mouse_callback(event, x, y, flags, param):
    global roi1, kcf_tracker, bytetrack_active, tracking_active
    global lost_frames, frame_buffer, bbox_history, center_history
    global fixed_roi_size, roi_adjust_frame_count
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if frame_buffer is None:
            print("프레임 버퍼 없음")
            return
        
        print(f"클릭: ({x}, {y}) - Dual Tracker 시작")
        
        half = 100
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        w = min(frame_buffer.shape[1] - x1, 2 * half)
        h = min(frame_buffer.shape[0] - y1, 2 * half)
        init_bbox = (x1, y1, w, h)
        
        kcf_tracker = cv2.legacy.TrackerKCF_create()
        success = kcf_tracker.init(frame_buffer, init_bbox)
        
        bytetrack_active = True
        tracking_active = True
        lost_frames = 0
        roi1 = init_bbox
        bbox_history = [(w, h)]
        center_history = [((x1 + w//2), (y1 + h//2))]
        fixed_roi_size = (int(w * roi_scale), int(h * roi_scale))
        roi_adjust_frame_count = 0
        
        if success:
            print(f"Dual Tracker 초기화: KCF {init_bbox}, ByteTrack ON")
        else:
            print(f"KCF init 실패: {init_bbox} - 그래도 시작")

cv2.namedWindow("Dual KCF + ByteTrack (Dynamic ROI1 v1.5)", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Dual KCF + ByteTrack (Dynamic ROI1 v1.5)", mouse_callback)

print("Dual KCF + ByteTrack (Dynamic ROI1 v1.5)")
print("ROI_SCALE=1.5 | Click:Track | B:Toggle | R:Reset | N:NextCam | Q:Quit")

while camera_manager.cap is not None and camera_manager.cap.isOpened():
    ret, frame = camera_manager.read_frame()
    if not ret or frame is None:
        print("프레임 읽기 실패")
        break
    
    frame_buffer = frame.copy()
    
    fps_end = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (fps_end - fps_start)
    fps_start = fps_end
    
    target_found = False
    
    if tracking_active and kcf_tracker is not None:
        success, bbox = kcf_tracker.update(frame)
        if success:
            x1, y1, w, h = map(int, bbox)
            roi1 = (x1, y1, w, h)
            
            bbox_history.append((w, h))
            if len(bbox_history) > BBOX_HISTORY_LEN:
                bbox_history.pop(0)
            avg_w = int(np.mean([wh[0] for wh in bbox_history]))
            avg_h = int(np.mean([wh[1] for wh in bbox_history]))
            fixed_roi_size = (int(avg_w * roi_scale), int(avg_h * roi_scale))
            
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 3)
            
            roi_crop = frame[y1:y1+h, x1:x1+w]
            bt_results = None
            if roi_crop.size > 0 and bytetrack_active:
                bt_results = model.track(
                    roi_crop, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD
                )[0]
            
            if bt_results is not None and bt_results.boxes is not None:
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
                    if confs[i] > CONF_THRESHOLD and dist < DIST_THRESHOLD:
                        score = dist * (1 - confs[i])
                        if score < min_dist:
                            min_dist = score
                            best_box = box
                            best_id = int(ids[i]) if ids is not None else -1
                
                if best_box is not None:
                    bx1, by1, bx2, by2 = map(int, best_box)
                    ox1, oy1, ox2, oy2 = x1+bx1, y1+by1, x1+bx2, y1+by2
                    
                    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
                    cv2.putText(
                        frame, f"ID:{best_id}", (ox1, oy1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                    )
                    
                    bbox_history.append((bx2-bx1, by2-by1))
                    if len(bbox_history) > BBOX_HISTORY_LEN:
                        bbox_history.pop(0)
                    
                    target_found = True
                    lost_frames = 0
                else:
                    lost_frames += 1
            else:
                lost_frames += 1
        else:
            lost_frames += 1

        roi_adjust_frame_count += 1
        if roi_adjust_frame_count >= roi_adjust_interval:
            roi_adjust_frame_count = 0
            
            if roi1 is not None and bytetrack_active:
                x1, y1, w, h = roi1
                roi_crop = frame[y1:y1+h, x1:x1+w]
                if roi_crop.size > 0:
                    high_conf_results = model.track(
                        roi_crop, persist=True, tracker="bytetrack.yaml",
                        conf=HIGH_CONF_THRESHOLD
                    )[0]
                    
                    if high_conf_results is not None and high_conf_results.boxes is not None:
                        boxes = high_conf_results.boxes.xyxy.cpu().numpy()
                        confs = high_conf_results.boxes.conf.cpu().numpy()
                        
                        best_high_conf_box = None
                        max_conf = 0
                        cx_crop, cy_crop = w//2, h//2
                        
                        for i, box in enumerate(boxes):
                            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                            dist = ((cx-cx_crop)**2 + (cy-cy_crop)**2)**0.5
                            if confs[i] > max_conf and dist < DIST_THRESHOLD * 0.5:
                                max_conf = confs[i]
                                best_high_conf_box = box
                        
                        if best_high_conf_box is not None:
                            bx1, by1, bx2, by2 = map(int, best_high_conf_box)
                            obj_w, obj_h = bx2-bx1, by2-by1
                            
                            obj_cx_crop = bx1 + obj_w//2
                            obj_cy_crop = by1 + obj_h//2
                            new_cx = x1 + obj_cx_crop
                            new_cy = y1 + obj_cy_crop
                            
                            new_w = int(obj_w * ROI_SCALE)
                            new_h = int(obj_h * ROI_SCALE)
                            new_x1 = max(0, new_cx - new_w//2)
                            new_y1 = max(0, new_cy - new_h//2)
                            new_x2 = min(frame.shape[1], new_x1 + new_w)
                            new_y2 = min(frame.shape[0], new_y1 + new_h)
                            new_roi1 = (new_x1, new_y1, new_x2-new_x1, new_y2-new_y1)
                            
                            kcf_tracker = cv2.legacy.TrackerKCF_create()
                            kcf_success = kcf_tracker.init(frame, new_roi1)
                            roi1 = new_roi1
                            
                            print(
                                f"ROI1 Adjust: {new_roi1} (conf:{max_conf:.2f}, "
                                f"obj_c:{int(new_cx)},{int(new_cy)}) {'OK' if kcf_success else 'FAIL'}"
                            )

    if tracking_active and roi1 is not None:
        if not target_found:
            cv2.rectangle(
                frame, (roi1[0], roi1[1]),
                (roi1[0]+roi1[2], roi1[1]+roi1[3]),
                (0, 165, 255), 2
            )
            cv2.putText(
                frame, f"Lost: {lost_frames}",
                (roi1[0], roi1[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2
            )
            
            if lost_frames > KEEP_FRAMES:
                reset_tracking_state()
                print("추적 실패 → IDLE")
        else:
            lost_frames = 0
    
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    status = "ByteTrack" if bytetrack_active else "KCF Only"
    color = (0, 255, 255) if bytetrack_active else (255, 255, 0)
    cv2.putText(
        frame, f"Dual: {status}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )
    
    if roi1:
        rw, rh = fixed_roi_size or (0, 0)
        cv2.putText(
            frame, f"Tgt:{rw}x{rh} Scl:{ROI_SCALE}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
        )
        cv2.putText(
            frame, f"Adj:{roi_adjust_frame_count}/{roi_adjust_interval}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 2
        )
    
    cv2.putText(
        frame, "Click:BTrack | B:Toggle | R:Reset | N:NextCam | Q:Quit",
        (10, 470),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 2
    )
    
    cv2.imshow("Dual KCF + ByteTrack (Dynamic ROI1 v1.5)", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        reset_tracking_state()
        print("리셋 완료")
    elif key == ord('b') or key == ord('B'):
        bytetrack_active = not bytetrack_active
        print(f"ByteTrack {'ON' if bytetrack_active else 'OFF'}")
    elif key == ord('n') or key == ord('N'):
        reset_tracking_state()
        camera_manager.switch_to_next()
        cap = camera_manager.cap
        if cap is None or not cap.isOpened():
            print("카메라 전환 후 열기 실패 → 종료")
            break

camera_manager.cleanup()
cv2.destroyAllWindows()
print("Dual KCF + ByteTrack (Dynamic ROI1 v1.5) 완료!")
print("ROI1=객체중심*1.5 | N키로 카메라 전환 테스트!")


