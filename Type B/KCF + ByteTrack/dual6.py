import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("/home/nes/yolo11n.engine", task="detect")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_buffer = None
kcf_tracker = None
bytetrack_active = False
tracking_active = False
roi1 = None
lost_frames = 0
KEEP_FRAMES = 30

roi_scale = 1.2
fixed_roi_size = None
center_history = []
bbox_history = []
BBOX_HISTORY_LEN = 5
HISTORY_LEN = 5
DIST_THRESHOLD = 100
CONF_THRESHOLD = 0.3

fps_start = cv2.getTickCount()

def mouse_callback(event, x, y, flags, param):
    global roi1, kcf_tracker, bytetrack_active, tracking_active, lost_frames, frame_buffer, bbox_history, center_history, fixed_roi_size
    
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
        
        if success:
            print(f"Dual Tracker 초기화: KCF {init_bbox}, ByteTrack ON")
        else:
            print(f"KCF init 실패: {init_bbox} - 그래도 시작")

cv2.namedWindow("Dual KCF + ByteTrack (Nested ROI)", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Dual KCF + ByteTrack (Nested ROI)", mouse_callback)

print("=== Dual KCF + ByteTrack (Nested ROI) ===")
print("Click: Dual Track | B: ByteTrack Toggle | R:Reset | Q:Quit")
print("ROI1(초록):KCF | ROI2(파랑,ID):ByteTrack")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
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
                bt_results = model.track(roi_crop, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD)[0]

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
                    cv2.putText(frame, f"ID:{best_id}", (ox1, oy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    bbox_history.append((bx2-bx1, by2-by1))
                    if len(bbox_history) > BBOX_HISTORY_LEN:
                        bbox_history.pop(0)
                    
                    target_found = True
                    lost_frames = 0
            else:
                lost_frames += 1
        else:
            lost_frames += 1

    if tracking_active and roi1 is not None:
        if not target_found:
            cv2.rectangle(frame, (roi1[0], roi1[1]), (roi1[0]+roi1[2], roi1[1]+roi1[3]), (0, 165, 255), 2)
            cv2.putText(frame, f"Lost: {lost_frames}", (roi1[0], roi1[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if lost_frames > KEEP_FRAMES:
                tracking_active = False
                kcf_tracker = None
                bytetrack_active = False
                roi1 = None
                fixed_roi_size = None
                center_history = []
                bbox_history = []
                print("추적 실패 → IDLE")
        else:
            lost_frames = 0

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    status = "ByteTrack" if bytetrack_active else "KCF Only"
    color = (0, 255, 255) if bytetrack_active else (255, 255, 0)
    cv2.putText(frame, f"Dual: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    if roi1:
        rw, rh = fixed_roi_size or (0,0)
        cv2.putText(frame, f"Target: {rw}x{rh}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    cv2.putText(frame, "Click:Dual | B:ByteTrack | R:Reset | Q:Quit", (10, 470), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow("Dual KCF + ByteTrack (Nested ROI)", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        tracking_active = False
        kcf_tracker = None
        bytetrack_active = False
        roi1 = None
        fixed_roi_size = None
        center_history = []
        bbox_history = []
        lost_frames = 0
        print("리셋 완료")
    elif key == ord('b') or key == ord('B'):
        bytetrack_active = not bytetrack_active
        print(f"ByteTrack {'ON' if bytetrack_active else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print("Dual KCF + ByteTrack 완료!")


