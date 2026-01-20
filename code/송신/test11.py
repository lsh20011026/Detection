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
target_track_id = None   # (트래커 제거, 일단 남겨둠)
ROI_W, ROI_H = 60, 60
frame_h, frame_w = 0, 0
tracking_mode = "NONE"   # "NONE" / "TEMPLATE"
yolo_enabled = False
last_cx, last_cy = None, None

# YOLO 재탐지 주기 (프레임 단위)
REDETECT_INTERVAL = 10    # 10프레임마다 ROI 안 객체 재탐지

# ===== 로그용 추가 변수 =====
lost_frame_count = 0      # 연속 LOW CONF 카운트
MAX_LOST_FRAMES = 30      # 추적 실패 기준

def mouse_callback(event, x, y, flags, param):
    global current_roi, template, target_track_id, tracking_mode
    global frame_h, frame_w, last_cx, last_cy, yolo_enabled, lost_frame_count

    win_name = "Hybrid_Tracking_Stable - FULLSCREEN"  # 윈도우 이름 참조

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

                    # 배경 비중 줄이기 위해 안쪽으로 약간 크롭
                    shrink = 0.1
                    w = x2 - x1
                    h = y2 - y1
                    x1 = int(x1 + w * shrink)
                    x2 = int(x2 - w * shrink)
                    y1 = int(y1 + h * shrink)
                    y2 = int(y2 - h * shrink)

                    current_roi = (x1, y1, x2, y2)
                    template = frame[y1:y2, x1:x2].copy()
                    tracking_mode = "TEMPLATE"
                    target_track_id = None
                    last_cx, last_cy = None, None
                    lost_frame_count = 0  # 로그 리셋
                    clicked_on_object = True
                    print(f"[YOLO→TEMPLATE] ROI set: {current_roi}")

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
                lost_frame_count = 0  # 로그 리셋
                print(f"[MANUAL TEMPLATE] ROI set: {current_roi}")

    # ===== 추가: 마우스 휠로 화면 크기 조절 =====
    elif event == cv2.EVENT_MOUSEWHEEL:
        rect = cv2.getWindowImageRect(win_name)
        w, h = rect[2], rect[3]
        if flags > 0:  # 휠 위로: 확대
            new_w, new_h = min(1920, w + 100), min(1080, h + 100)
            cv2.resizeWindow(win_name, new_w, new_h)
            print(f"ZOOM IN: {new_w}x{new_h}")
        else:  # 휠 아래로: 축소
            new_w, new_h = max(640, w - 100), max(480, h - 100)
            cv2.resizeWindow(win_name, new_w, new_h)
            print(f"ZOOM OUT: {new_w}x{new_h}")

def main(video_path="/home/nes/cctv.mp4"):
    global current_roi, template, target_track_id, tracking_mode
    global frame_h, frame_w, last_cx, last_cy, yolo_enabled
    global lost_frame_count

    try:
        ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)
        print("✅ Serial enabled")
    except:
        ser = None
        print("⚠️  Serial disabled")

    # ===================== TENSORRT 엔진으로 교체 =====================
    # 기존: model = YOLO("yolo11n.pt")
    model = YOLO("/home/nes/yolo11n.engine", task='detect')
    print("🚀 TensorRT YOLO Engine Loaded!")
    # ==================================================================

    cap = cv2.VideoCapture(video_path)
    print(f"📹 Video mode: {video_path}")

    if not cap.isOpened():
        print("❌ Capture failed")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"🖥️  Resolution: {actual_w}x{actual_h}")

    # ===== 큰 화면으로 시작 =====
    win_name = "Hybrid_Tracking_Stable - FULLSCREEN"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)  # 크게 시작!
    # 전체화면 원하면 아래 주석 해제:
    # cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mouse_param = {"frame": None, "boxes": None}
    cv2.setMouseCallback(win_name, mouse_callback, mouse_param)

    frame_count = 0
    prev_time = time.time()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 Total frames: {total_frames} | Controls: t=YOLO, r=RESET, q=QUIT, 마우스휠=ZOOM")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🏁 End of video")
            break

        frame_h, frame_w = frame.shape[:2]
        mouse_param["frame"] = frame
        frame_count += 1

        # 진행률
        if total_frames > 0:
            progress = frame_count / total_frames * 100
            cv2.putText(frame, f"F:{frame_count}/{total_frames} ({progress:.1f}%)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ===== 1) 템플릿 매칭 (기본 추적) + 상세 로그 =====
        if current_roi is not None and tracking_mode == "TEMPLATE" and template is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                th, tw = tpl_gray.shape[:2]

                rx1, ry1, rx2, ry2 = current_roi
                roi_cx, roi_cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
                
                margin = 80
                sx1 = max(0, rx1 - margin)
                sy1 = max(0, ry1 - margin)
                sx2 = min(frame_w, rx2 + margin)
                sy2 = min(frame_h, ry2 + margin)

                if (sx2 - sx1) > tw and (sy2 - sy1) > th:
                    search_roi = gray[sy1:sy2, sx1:sx2]
                    res = cv2.matchTemplate(search_roi, tpl_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)

                    # ===== 상세 로그 =====
                    new_cx = max_loc[0] + sx1
                    new_cy = max_loc[1] + sy1
                    drift_dist = ((new_cx - roi_cx)**2 + (new_cy - roi_cy)**2)**0.5
                    search_w, search_h = sx2 - sx1, sy2 - sy1
                    
                    print(f"F{frame_count:4d} | TMP:{max_val:.3f} | "
                          f"ROI({rx1:3d},{ry1:3d},{rx2:3d},{ry2:3d})→"
                          f"NEW({new_cx:.1f},{new_cy:.1f}) | "
                          f"DRIFT:{drift_dist:.1f}px | "
                          f"SRCH:{search_w:3d}x{search_h:3d}")

                    # 상관값이 충분히 높을 때만 ROI + 템플릿 갱신
                    if max_val > 0.7:
                        x1 = max_loc[0] + sx1
                        y1 = max_loc[1] + sy1
                        x2, y2 = x1 + tw, y1 + th
                        current_roi = (max(0, x1), max(0, y1),
                                       min(frame_w - 1, x2), min(frame_h - 1, y2))
                        template = frame[int(y1):int(y2), int(x1):int(x2)].copy()
                        
                        print(f"     ✓ UPDATE OK (conf={max_val:.3f}, drift={drift_dist:.1f}px)")
                        lost_frame_count = 0
                        cv2.putText(frame, f"TEMPLATE {max_val:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        lost_frame_count += 1
                        print(f"     ✗ LOW CONF! ({max_val:.3f}) LOST:{lost_frame_count}/{MAX_LOST_FRAMES}")
                        
                        if lost_frame_count >= 5:
                            print(f"     ⚠️  연속 LOW CONF {lost_frame_count}프레임!")
                        if drift_dist > 50:
                            print(f"     🚨 DRIFT DETECTED! {drift_dist:.1f}px 급변!")
                            
                        cv2.putText(frame, f"TEMPLATE LOW {max_val:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

                else:
                    print(f"F{frame_count:4d} | ❌ SEARCH AREA TOO SMALL! ({sx2-sx1}x{sy2-sy1})")
                    
            except Exception as e:
                print(f"F{frame_count:4d} | 💥 TEMPLATE ERROR: {e}")

        # ===== 2) YOLO 탐지 + 로그 (TensorRT 엔진으로 실행) =====
        mouse_param["boxes"] = None

        if yolo_enabled:
            try:
                results = model.predict(
                    source=frame,
                    device=0,
                    verbose=False,
                    conf=0.35,
                    imgsz=320,
                    max_det=5
                )

                for r in results:
                    boxes = r.boxes
                    if boxes is None:
                        continue

                    mouse_param["boxes"] = boxes

                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        class_name = model.names[cls_id]

                        cx = (xyxy[0] + xyxy[2]) / 2
                        cy = (xyxy[1] + xyxy[3]) / 2

                        cv2.rectangle(frame,
                                      (int(xyxy[0]), int(xyxy[1])),
                                      (int(xyxy[2]), int(xyxy[3])),
                                      (128, 128, 128), 1)

                # YOLO 재탐지 로그
                if current_roi is not None and (frame_count % REDETECT_INTERVAL == 0):
                    rx1, ry1, rx2, ry2 = current_roi
                    roi_cx = (rx1 + rx2) / 2
                    roi_cy = (ry1 + ry2) / 2

                    best_box = None
                    best_score = -1
                    best_conf = 0

                    if mouse_param["boxes"] is not None:
                        for box in mouse_param["boxes"]:
                            xyxy = box.xyxy[0].tolist()
                            cx = (xyxy[0] + xyxy[2]) / 2
                            cy = (xyxy[1] + xyxy[3]) / 2
                            conf = float(box.conf[0])

                            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                                dist2 = (cx - roi_cx)**2 + (cy - roi_cy)**2
                                score = -dist2

                                if score > best_score:
                                    best_score = score
                                    best_box = xyxy
                                    best_conf = conf

                    if best_box is not None:
                        x1, y1, x2, y2 = map(int, best_box)
                        shrink = 0.1
                        w = x2 - x1
                        h = y2 - y1
                        x1 = int(x1 + w * shrink)
                        x2 = int(x2 - w * shrink)
                        y1 = int(y1 + h * shrink)
                        y2 = int(y2 - h * shrink)

                        old_roi = current_roi
                        current_roi = (x1, y1, x2, y2)
                        template = frame[y1:y2, x1:x2].copy()
                        
                        print(f"F{frame_count:4d} | [REDETECT✓] conf={best_conf:.3f} "
                              f"ROI:{old_roi}→{current_roi} | score:{best_score:.1f}")
                        lost_frame_count = 0
                    else:
                        print(f"F{frame_count:4d} | [REDETECT✗] ROI({rx1},{ry1},{rx2},{ry2}) 내 객체 없음")

            except Exception as e:
                print(f"F{frame_count:4d} | YOLO error: {e}")

        # ===== 3) ROI 시각화 =====
        if current_roi is not None:
            x1, y1, x2, y2 = current_roi
            roi_color = (0, 255, 255)
            roi_text = "TEMPLATE"
            cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 3)
            cv2.putText(frame, roi_text, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)

        # 상태 표시 (LOST 카운트 추가)
        status = f"M:{tracking_mode[:4]} Y:{'ON' if yolo_enabled else 'OFF'} L:{lost_frame_count}"
        cv2.putText(frame, status, (10, frame_h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "t:YOLO r:reset q:quit Wheel:ZOOM", (10, frame_h - 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
            print(f"F{frame_count:4d} | ===== MANUAL RESET! ====")
            current_roi = None
            template = None
            target_track_id = None
            tracking_mode = "NONE"
            last_cx, last_cy = None, None
            lost_frame_count = 0
        elif key == ord('t'):
            yolo_enabled = not yolo_enabled
            print(f"F{frame_count:4d} | YOLO {'ON' if yolo_enabled else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()
    print("👋 Tracking ended")

if __name__ == "__main__":
    main("/home/nes/Timeline1.mov")




