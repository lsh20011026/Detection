import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        params = param
        cx, cy = x, y
        half = 40
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(params['frame_shape'][1], cx + half)
        y2 = min(params['frame_shape'][0], cy + half)
        w = x2 - x1
        h = y2 - y1
        params['init_bbox'] = (x1, y1, w, h)
        params['tracking_init'] = True
        print(f"ROI set: {params['init_bbox']}")

params = {
    'frame_shape': None,
    'init_bbox': None,
    'tracking_init': False,
    'tracker': None
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

ret, frame = cap.read()
if ret:
    params['frame_shape'] = frame.shape

cv2.namedWindow('KCF Tracker')
cv2.setMouseCallback('KCF Tracker', mouse_callback, params)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (curr_time - prev_time) if prev_time > 0 else 30.0
    prev_time = curr_time

    if params['tracking_init'] and params['init_bbox'] is not None:
        if params['tracker'] is not None:
            params['tracker'] = None
        params['tracker'] = cv2.TrackerKCF_create()
        params['tracker'].init(frame, params['init_bbox'])
        params['tracking_init'] = False
        print("KCF tracker initialized")

    if params['tracker'] is not None:
        success, bbox = params['tracker'].update(frame)
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failed - click to restart", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (50, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Click: Start tracking / r: Reset / q: Quit", 
                (50, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('KCF Tracker', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        params['tracker'] = None
        params['init_bbox'] = None

cap.release()
cv2.destroyAllWindows()


