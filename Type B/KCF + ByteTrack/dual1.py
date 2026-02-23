import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("/home/nes/yolo11n.engine")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

mode = 0
frame_buffer = None

target_center = None
target_roi = None
fixed_roi_size = None
center_history = []
roi_margin = 10
tracking_active = False
fps_start = cv2.getTickCount()
lost_frames = 0

tracker = None
KCF_MODE = 2
KEEP_FRAMES = 30

DIST_THRESHOLD = 100
CONF_THRESHOLD = 0.3
HISTORY_LEN = 5

def mouse_callback(event, x, y, flags, param):
    global target_center, target_roi, fixed_roi_size, center_history, tracking_active, lost_frames, mode, frame_buffer, tracker
    if event == cv2.EVENT_LBUTTONDOWN:
        if frame_buffer is None:
            print("Frame buffer empty")
            return
        
        print(f"Click: ({x}, {y}) | mode={mode}")
        target_center = (x, y)
        
        if mode == 0:
            half = 40
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(frame_buffer.shape[1], x + half)
            y2 = min(frame_buffer.shape[0], y + half)
            init_bbox = (x1, y1, x2-x1, y2-y1)
            
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame_buffer, init_bbox)
            mode = KCF_MODE
            tracking_active = True
            lost_frames = 0
            print(f"KCF initialized: {init_bbox}")
            
        else:
            fixed_roi_size = None
            center_history = [(x, y)]
            tracking_active = True
            lost_frames = 0
            print("YOLO object tracking started")
        
        target_roi = None

cv2.namedWindow("YOLO + KCF Tracker", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("YOLO + KCF Tracker", mouse_callback)

print("=== YOLO + KCF integrated tracker ===")
print("T: YOLO toggle | Click: Track | R:Reset | Q:Quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_buffer = frame.copy()
    
    fps_end = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (fps_end - fps_start)
    fps_start = fps_end
    
    results = None
    if mode == 1:
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25)[0]
    
    target_found = False
    
    if tracking_active:
        if mode == KCF_MODE and tracker is not None:
            success, bbox = tracker.update(frame)
            if success:
                target_roi = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 0), 3)
                lost_frames = 0
                target_found = True
            else:
                lost_frames += 1
        
        elif results is not None and results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            
            min_dist = float('inf')
            best_box = None
            best_cx, best_cy = None, None
            
            for i, box in enumerate(boxes):
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                dist = ((cx-target_center[0])**2 + (cy-target_center[1])**2)**0.5
                
                if confs[i] > CONF_THRESHOLD and dist < DIST_THRESHOLD:
                    score = dist * (1 - confs[i])
                    if score < min_dist:
                        min_dist = score
                        best_box = box
                        best_cx, best_cy = int(cx), int(cy)
            
            if best_box is not None:
                target_found = True
                center_history.append((best_cx, best_cy))
                if len(center_history) > HISTORY_LEN:
                    center_history.pop(0)
                
                avg_cx = int(np.mean([c[0] for c in center_history]))
                avg_cy = int(np.mean([c[1] for c in center_history]))
                target_center = (avg_cx, avg_cy)
                
                if fixed_roi_size is None:
                    x1, y1, x2, y2 = map(int, best_box)
                    rw = min(frame.shape[1], x2 + roi_margin) - max(0, x1 - roi_margin)
                    rh = min(frame.shape[0], y2 + roi_margin) - max(0, y1 - roi_margin)
                    fixed_roi_size = (rw, rh)
                
                rw, rh = fixed_roi_size
                roi_x1 = max(0, avg_cx - rw//2)
                roi_y1 = max(0, avg_cy - rh//2)
                roi_x2 = min(frame.shape[1], avg_cx + rw//2)
                roi_y2 = min(frame.shape[0], avg_cy + rh//2)
                target_roi = (roi_x1, roi_y1, roi_x2, roi_y2)
                
                cv2.rectangle(frame, target_roi, (0, 255, 0), 3)
    
    if tracking_active and target_roi is not None:
        if not target_found:
            lost_frames += 1
            cv2.rectangle(frame, target_roi, (0, 165, 255), 2)
            cv2.putText(frame, f"Lost: {lost_frames}", (target_roi[0], target_roi[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if lost_frames > KEEP_FRAMES:
                tracking_active = False
                target_roi = None
                fixed_roi_size = None
                center_history = []
                tracker = None
                mode = 0
                print("Tracking failed -> IDLE mode")
        else:
            lost_frames = 0
    
    if mode == 1 and results is not None:
        annotated = results.plot()
        annotated[frame.shape[0]-40:frame.shape[0], :] = frame[frame.shape[0]-40:frame.shape[0], :]
    else:
        annotated = frame
    
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    mode_names = ["IDLE(KCF)", "YOLO", "KCF"]
    status_color = (0, 255, 255) if mode == 1 else (255, 255, 0)
    cv2.putText(annotated, mode_names[mode], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(annotated, "T:YOLO | Click:Track | R:Reset | Q:Quit", (10, 480-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("YOLO + KCF Tracker", annotated)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        target_center = None
        target_roi = None
        fixed_roi_size = None
        center_history = []
        tracking_active = False
        lost_frames = 0
        tracker = None
        mode = 0
        print("Full reset")
    elif key == ord('t') or key == ord('T'):
        if mode == 0:
            mode = 1
            print("YOLO detect ON")
        elif mode == 1:
            mode = 0
            print("IDLE(KCF ready)")

cap.release()
cv2.destroyAllWindows()
print("YOLO + KCF integration complete")


