import cv2
import numpy as np
import time
from ultralytics import YOLO

model = YOLO('/home/nes/yolo11n.engine') 

tracker = None
tracking_active = False
selected_bbox = None      # 클릭해서 선택한 실제 target bbox (전역 좌표)
track_roi = None          # 🔥 트래킹에 사용할 주변 ROI (전역 좌표)
bboxes = []
frame_copy = None

prev_time = 0
ROI_PAD = 80  # 🔥 선택한 bbox 주변 여유 픽셀 (60~120 추천)

def mouse_callback(event, x, y, flags, param):
    global selected_bbox, tracking_active, tracker, frame_copy, track_roi

    if event == cv2.EVENT_LBUTTONDOWN:
        # 🔥 1단계: YOLO bbox 내부 클릭 여부만 확인 (가장 가까운 찾기 제거)
        selected_bbox = None
        for bbox in bboxes:
            bx, by, bw, bh = bbox[:4]
            if bx <= x <= bx + bw and by <= y <= by + bh:  # 내부 클릭만!
                selected_bbox = (int(bx), int(by), int(bw), int(bh))
                print(f"YOLO bbox 선택: {selected_bbox}")
                break

        # 2단계: bbox 내부 아니면 클릭 위치 중심 ROI 생성
        if selected_bbox is None:
            roi_size = 120
            half = roi_size // 2
            sx = max(0, x - half)
            sy = max(0, y - half)
            ex = min(frame_copy.shape[1], x + half)
            ey = min(frame_copy.shape[0], y + half)
            selected_bbox = (int(sx), int(sy), int(ex - sx), int(ey - sy))
            print(f"ROI 생성 (클릭: {x},{y}): {selected_bbox}")

        # 🔥 3단계: track_roi 생성 + tracker ROI로만 init
        if selected_bbox:
            x, y, w, h = selected_bbox
            rx1 = max(0, x - ROI_PAD)
            ry1 = max(0, y - ROI_PAD)
            rx2 = min(frame_copy.shape[1], x + w + ROI_PAD)
            ry2 = min(frame_copy.shape[0], y + h + ROI_PAD)
            track_roi = (rx1, ry1, rx2 - rx1, ry2 - ry1)
            print(f"Track ROI: {track_roi} (PAD={ROI_PAD})")

            # ROI 이미지 추출 + 로컬 좌표 변환
            roi_frame = frame_copy[ry1:ry1 + track_roi[3], rx1:rx1 + track_roi[2]].copy()
            local_bbox = (x - rx1, y - ry1, w, h)

            tracker = cv2.TrackerCSRT_create()
            tracker.init(roi_frame, local_bbox)
            tracking_active = True

cap = cv2.VideoCapture(0)
cv2.namedWindow('YOLO + CSRT Tracker', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('YOLO + CSRT Tracker', mouse_callback)

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
        # 🔥 tracker.update도 ROI로만!
        if track_roi is None:
            tracking_active = False
            tracker = None
        else:
            roi_frame = frame_copy[track_roi[1]:track_roi[1] + track_roi[3], 
                                  track_roi[0]:track_roi[0] + track_roi[2]]
            
            success, local_bbox = tracker.update(roi_frame)
            if success:
                # 로컬 → 전역 좌표 복원
                gx = int(local_bbox[0] + track_roi[0])
                gy = int(local_bbox[1] + track_roi[1])
                gw = int(local_bbox[2])
                gh = int(local_bbox[3])
                
                cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (0, 255, 0), 2)
                cv2.putText(frame, f'Tracking (ROI:{track_roi[2]}x{track_roi[3]})', 
                           (gx, gy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Tracking Fail', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                tracking_active = False
                tracker = None
                selected_bbox = None
                track_roi = None

    # 🔥 ROI 영역 시각화 (파란색 사각형) - mouse_callback에서 설정된 track_roi 표시
    if tracking_active and track_roi is not None:
        rx1, ry1, rw, rh = track_roi
        cv2.rectangle(frame, (rx1, ry1), (rx1+rw, ry1+rh), (255, 0, 0), 2)
        cv2.putText(frame, f'ROI:{rw}x{rh}', (rx1, ry1-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('YOLO + CSRT Tracker', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') or key == ord('R'):
        print("R 키: 트래커 / ROI 초기화")
        tracking_active = False
        tracker = None
        selected_bbox = None
        track_roi = None

cap.release()
cv2.destroyAllWindows()


