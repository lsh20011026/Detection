import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("/home/nes/yolo11n.engine")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

DIST_THRESHOLD = 100
CONF_THRESHOLD = 0.3
HISTORY_LEN = 5
KEEP_FRAMES = 30

def mouse_callback(event, x, y, flags, param):
    global target_center, target_roi, fixed_roi_size, center_history, tracking_active, lost_frames, mode, frame_buffer
    if event == cv2.EVENT_LBUTTONDOWN:
        if frame_buffer is None:
            print("Frame buffer empty")
            return
        
        print(f"Click detected: ({x}, {y}) | mode={mode}")
        
        target_center = (x, y)
        
        if mode == 0:
            fixed_roi_size = (80, 80)
            print("Base mode: 80x80 fixed ROI")
            mode = 2
        else:
            fixed_roi_size = None
            print("Detect mode: bbox based ROI")
        
        center_history = [(x, y)]
        tracking_active = True
        lost_frames = 0

cv2.namedWindow("Mode Toggle ROI - FIXED", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Mode Toggle ROI - FIXED", mouse_callback)

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
    if tracking_active and target_center is not None:
        if results is not None and results.boxes is not None:
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
            
            roi_x1, roi_y1, roi_x2, roi_y2 = target_roi
            
            if lost_frames <= KEEP_FRAMES:
                cv2.rectangle(frame, target_roi, (0, 255, 0), 3)
                cv2.putText(frame, f"hold:{lost_frames}", (roi_x1, roi_y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                tracking_active = False
                target_roi = None
                fixed_roi_size = None
                mode = 0
                print("Reset after 1s -> base mode")
        else:
            lost_frames = 0
    
    if not tracking_active or target_roi is None:
        if mode == 1 and results is not None:
            annotated = results.plot()
        else:
            annotated = frame.copy()
    else:
        annotated = frame.copy()

    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    mode_names = ["IDLE", "DETECT", "TRACK"]
    status_color = (0, 255, 255) if mode == 1 else (255, 255, 0)
    cv2.putText(annotated, mode_names[mode], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(annotated, "T:Toggle Detect | Click→ROI 1s | R:Reset | Q:Quit", (10, 480-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Mode Toggle ROI - FIXED", annotated)
    
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
        mode = 0
        print("Full reset")
    elif key == ord('t') or key == ord('T'):
        if mode == 0:
            mode = 1
            print("Detect mode ON")
        elif mode == 1:
            mode = 0
            print("Base mode OFF")

cap.release()
cv2.destroyAllWindows()
print("Mode Toggle + click working")


