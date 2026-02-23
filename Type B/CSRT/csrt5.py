import cv2
import numpy as np
import time
from ultralytics import YOLO

# 🔥 IoU 계산 함수 (재초기화용)
def calculate_iou(boxA, boxB):
    # box: (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

model = YOLO('/home/nes/yolo11n.engine') 

tracker = None
tracking_active = False
selected_bbox = None
track_roi = None
bboxes = []
frame_copy = None

prev_time = 0
ROI_PAD = 80
ROI_EXPAND = 15
MAX_ROI_SIZE = 350

def mouse_callback(event, x, y, flags, param):
    global selected_bbox, tracking_active, tracker, frame_copy, track_roi

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_bbox = None
        for bbox in bboxes:
            bx, by, bw, bh = bbox[:4]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                selected_bbox = (int(bx), int(by), int(bw), int(bh))
                print(f"YOLO bbox 선택: {selected_bbox}")
                break

        if selected_bbox is None:
            roi_size = 120
            half = roi_size // 2
            sx = max(0, x - half)
            sy = max(0, y - half)
            ex = min(frame_copy.shape[1], x + half)
            ey = min(frame_copy.shape[0], y + half)
            selected_bbox = (int(sx), int(sy), int(ex - sx), int(ey - sy))
            print(f"ROI 생성 (클릭: {x},{y}): {selected_bbox}")

        if selected_bbox:
            x, y, w, h = selected_bbox
            rx1 = max(0, x - ROI_PAD)
            ry1 = max(0, y - ROI_PAD)
            rx2 = min(frame_copy.shape[1], x + w + ROI_PAD)
            ry2 = min(frame_copy.shape[0], y + h + ROI_PAD)
            track_roi = (rx1, ry1, rx2 - rx1, ry2 - ry1)
            print(f"Track ROI: {track_roi}")

            roi_frame = frame_copy[ry1:ry1 + track_roi[3], rx1:rx1 + track_roi[2]].copy()
            local_bbox = (x - rx1, y - ry1, w, h)

            tracker = cv2.TrackerCSRT_create()
            tracker.init(roi_frame, local_bbox)
            tracking_active = True

cap = cv2.VideoCapture(0)
cv2.namedWindow('YOLO + CSRT Tracker v3', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('YOLO + CSRT Tracker v3', mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    
    frame_copy = frame.copy()
    
    if not tracking_active:
        results = model(frame, conf=0.5)
        bboxes = []
        for r in results:
            boxes = r.boxes.xywh.cpu().numpy() if r.boxes is not None else np.array([])
            for box in boxes:
                x, y, w, h = box[:4]
                conf = box[4] if len(box) > 4 else 0.5
                bboxes.append((int(x-w/2), int(y-h/2), int(w), int(h), conf, 0))
        
        for bbox in bboxes:
            x, y, w, h, conf, cls = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f'{conf:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    else:
        # 🔥 최종 버전: ROI 제한 + 자동 재초기화
        if track_roi is None:
            tracking_active = False
            tracker = None
        else:
            rx1, ry1, rw, rh = track_roi
            roi_frame = frame_copy[ry1:ry1 + rh, rx1:rx1 + rw]
            
            success, local_bbox = tracker.update(roi_frame)
            if success:
                gx = int(local_bbox[0] + rx1)
                gy = int(local_bbox[1] + ry1)
                gw = int(local_bbox[2])
                gh = int(local_bbox[3])
                
                # 🔥 ROI 크기 제한 + 동적 업데이트
                new_rx1 = max(0, gx - ROI_PAD - ROI_EXPAND)
                new_ry1 = max(0, gy - ROI_PAD - ROI_EXPAND)
                new_rx2 = min(frame_copy.shape[1], gx + gw + ROI_PAD + ROI_EXPAND)
                new_ry2 = min(frame_copy.shape[0], gy + gh + ROI_PAD + ROI_EXPAND)
                
                new_rw = min(new_rx2 - new_rx1, MAX_ROI_SIZE)
                new_rh = min(new_ry2 - new_ry1, MAX_ROI_SIZE)
                new_rx1 = max(0, new_rx2 - new_rw)
                new_ry1 = max(0, new_ry2 - new_rh)
                
                track_roi = (int(new_rx1), int(new_ry1), int(new_rw), int(new_rh))
                
                cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (0, 255, 0), 3)
                cv2.putText(frame, f'Track ROI:{track_roi[2]}x{track_roi[3]}', 
                           (gx, gy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                           
            else:
                # 🔥 트래킹 실패 → ROI 내 YOLO 재검색
                print("트래킹 실패 → YOLO 재검색")
                roi_results = model(roi_frame, conf=0.4)
                best_match = None
                best_iou = 0
                prev_global_box = (gx, gy, gx+gw, gy+gh)
                
                for r in roi_results:
                    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.array([])
                    for box in boxes:
                        bx1, by1, bx2, by2 = box[:4]
                        # ROI 내 bbox만 (로컬 좌표)
                        if 0 <= bx1 and bx2 <= rw and 0 <= by1 and by2 <= rh:
                            local_box = (bx1, by1, bx2, by2)
                            global_box = (bx1+rx1, by1+ry1, bx2+rx1, by2+ry1)
                            iou = calculate_iou(prev_global_box, global_box)
                            if iou > best_iou and iou > 0.3:
                                best_iou = iou
                                best_match = local_box
                
                if best_match:
                    tracker.init(roi_frame, best_match)
                    gx = int(best_match[0] + rx1)
                    gy = int(best_match[1] + ry1)
                    gw = int(best_match[2])
                    gh = int(best_match[3])
                    print(f"✅ YOLO 재초기화! IoU:{best_iou:.2f}")
                    cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (0, 255, 255), 3)  # 노란색
                else:
                    cv2.putText(frame, 'Tracking Fail', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    tracking_active = False
                    tracker = None
                    selected_bbox = None
                    track_roi = None

    # ROI 시각화
    if tracking_active and track_roi is not None:
        rx1, ry1, rw, rh = track_roi
        cv2.rectangle(frame, (rx1, ry1), (rx1+rw, ry1+rh), (255, 0, 0), 2)
        cv2.putText(frame, f'ROI:{rw}x{rh}', (rx1, ry1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('YOLO + CSRT Tracker v3', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') or key == ord('R'):
        print("R 키: 트래커 초기화")
        tracking_active = False
        tracker = None
        selected_bbox = None
        track_roi = None

cap.release()
cv2.destroyAllWindows()


