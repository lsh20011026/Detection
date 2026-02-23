import cv2
import numpy as np
import time
from ultralytics import YOLO
import serial
import sys
sys.path.append("/home/nes/.local/lib/python3.10/site-packages")

# ===== 전역 상태 =====
current_roi = None
template = None
target_track_id = None   # 트래커 제거했지만 일단 남겨둠
ROI_W, ROI_H = 50, 50
frame_h, frame_w = 0, 0
tracking_mode = "NONE"   # "NONE" / "TEMPLATE"
yolo_enabled = False
last_cx, last_cy = None, None

def mouse_callback(event, x, y, flags, param):
    global current_roi, template, target_track_id, tracking_mode
    global frame_h, frame_w, last_cx, last_cy, yolo_enabled

    if event == cv2.EVENT_LBUTTONDOWN:
        boxes = param["boxes"]
        frame = param["frame"]

        clicked_on_object = False

        # 1) YOLO BBOX 위 클릭 시: 그 박스를 템플릿 ROI로 사용
        if boxes is not None and yolo_enabled:
            for box in boxes:
                b_xyxy = box.xyxy[0].tolist()
                if (b_xyxy[0] <= x <= b_xyxy[2] and b_xyxy[1] <= y <= b_xyxy[3]):
                    x1 = int(b_xyxy[0])
                    y1 = int(b_xyxy[1])
                    x2 = int(b_xyxy[2])
                    y2 = int(b_xyxy[3])

                    current_roi = (x1, y1, x2, y2)
                    template = frame[y1:y2, x1:x2].copy()
                    tracking_mode = "TEMPLATE"
                    target_track_id = None
                    last_cx, last_cy = None, None
                    clicked_on_object = True
                    print(f"[YOLO→TEMPLATE] ROI set from box: {current_roi}")

                    # 필요하면 여기서 YOLO 끄기
                    # yolo_enabled = False
                    # print("YOLO OFF → TEMPLATE-only tracking")

                    break

        # 2) 빈 공간 클릭: 고정 크기 템플릿 ROI
        if not clicked_on_object:
            x1 = max(0, int(x - ROI_W / 2))
            y1 = max(0, int(y - ROI_H / 2))
            x2 = min(frame_w - 1, int(x + ROI_W / 2))
            y2 = min(frame_h - 1, int(y + ROI_H / 2))
            if x2 > x1 and y2 > y1:
                current_roi = (x1, y1, x2, y2)
                template = frame[y1:y2, x1:x2].copy()
                tracking_mode = "TEMPLATE"
                target_track_id = None
                last_cx, last_cy = None, None
                print(f"[MANUAL TEMPLATE] ROI set: {current_roi}")


def main(video_path="/home/nes/cctv.mp4"):
    global current_roi, template, target_track_id, tracking_mode
    global frame_h, frame_w, last_cx, last_cy, yolo_enabled

    try:
        ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)
        print("Serial enabled")
    except:
        ser = None
        print("Serial disabled")

    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(video_path)
    print(f"Video mode: {video_path}")

    if not cap.isOpened():
        print("Capture failed")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {actual_w}x{actual_h}")

    win_name = "Hybrid_Tracking_Stable"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    mouse_param = {"frame": None, "boxes": None}
    cv2.setMouseCallback(win_name, mouse_callback, mouse_param)

    frame_count = 0
    prev_time = time.time()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        frame_h, frame_w = frame.shape[:2]
        mouse_param["frame"] = frame
        frame_count += 1

        # 진행률
        if total_frames > 0:
            progress = frame_count / total_frames * 100
            cv2.putText(frame, f"F:{frame_count}/{total_frames} ({progress:.1f}%)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ===== 템플릿 매칭 (기본 모드) =====
        if current_roi is not None and tracking_mode == "TEMPLATE" and template is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                th, tw = tpl_gray.shape[:2]

                rx1, ry1, rx2, ry2 = current_roi
                margin = 80
                sx1 = max(0, rx1 - margin)
                sy1 = max(0, ry1 - margin)
                sx2 = min(frame_w, rx2 + margin)
                sy2 = min(frame_h, ry2 + margin)

                if (sx2 - sx1) > tw and (sy2 - sy1) > th:
                    search_roi = gray[sy1:sy2, sx1:sx2]
                    res = cv2.matchTemplate(search_roi, tpl_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)

                    if max_val > 0.6:
                        x1 = max_loc[0] + sx1
                        y1 = max_loc[1] + sy1
                        x2, y2 = x1 + tw, y1 + th
                        current_roi = (max(0, x1), max(0, y1),
                                       min(frame_w - 1, x2), min(frame_h - 1, y2))
                        # 템플릿 업데이트
                        template = frame[y1:y2, x1:x2].copy()
                        cv2.putText(frame, f"TEMPLATE {max_val:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            except:
                pass

        # ===== YOLO: 탐지 전용 (t로 ON/OFF) =====
        mouse_param["boxes"] = None
        if yolo_enabled:
            try:
                results = model.predict(
                    source=frame,
                    device=0,
                    verbose=False,
                    conf=0.35,
                    imgsz=416,
                    max_det=5
                )

                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        mouse_param["boxes"] = boxes

                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].tolist()
                            class_name = model.names[cls_id]

                            # 중심 좌표 (지금은 클릭 판정용만)
                            cx = (xyxy[0] + xyxy[2]) / 2
                            cy = (xyxy[1] + xyxy[3]) / 2

                            # YOLO 탐지 시각화(얇은 회색만)
                            cv2.rectangle(frame,
                                          (int(xyxy[0]), int(xyxy[1])),
                                          (int(xyxy[2]), int(xyxy[3])),
                                          (128, 128, 128), 1)

                            # 필요시 YOLO 정보 시리얼 전송 가능
                            # if ser:
                            #     data = f"YOLO,{class_name},{conf:.2f},{xyxy}\n"
                            #     ser.write(data.encode())

            except Exception as e:
                print(f"YOLO error: {e}")

        # ===== ROI 시각화 =====
        if current_roi is not None:
            x1, y1, x2, y2 = current_roi
            roi_color = (0, 255, 255)  # TEMPLATE 모드 노랑
            roi_text = "TEMPLATE"
            cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 3)
            cv2.putText(frame, roi_text, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)

        # 상태 표시
        status = f"M:{tracking_mode[:4]} Y:{'ON' if yolo_enabled else 'OFF'}"
        cv2.putText(frame, status, (10, frame_h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "t:YOLO r:reset q:quit",
                    (10, frame_h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if frame_count > 1 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS:{fps:.1f}", (10, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            current_roi = None
            template = None
            target_track_id = None
            tracking_mode = "NONE"
            last_cx, last_cy = None, None
            print("RESET")
        elif key == ord('t'):
            yolo_enabled = not yolo_enabled
            print(f"YOLO {'ON' if yolo_enabled else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()

if __name__ == "__main__":
    main("/home/nes/Timeline1.mov")




