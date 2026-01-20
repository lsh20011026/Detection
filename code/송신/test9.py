import cv2
import time
from ultralytics import YOLO
import serial
import sys
sys.path.append("/home/nes/.local/lib/python3.10/site-packages")

# ====== 전역 상태 ======
current_roi = None        # (x1, y1, x2, y2)
ROI_W = 200               # ROI 가로
ROI_H = 200               # ROI 세로

last_cx = None            # 직전 타겟 중심 x
last_cy = None            # 직전 타겟 중심 y

MAX_JUMP_DIST2 = 200**2   # 한 프레임에서 허용할 최대 이동 거리 제곱 (px^2)
LERP_ALPHA = 0.3          # 새 위치로 몇 %만 따라갈지 (0~1)


def mouse_callback(event, x, y, flags, param):
    global current_roi, ROI_W, ROI_H, last_cx, last_cy

    if event == cv2.EVENT_LBUTTONDOWN:
        cx, cy = x, y
        frame_h, frame_w = param["frame_shape"]  # (height, width)

        x1 = int(cx - ROI_W / 2)
        y1 = int(cy - ROI_H / 2)
        x2 = int(cx + ROI_W / 2)
        y2 = int(cy + ROI_H / 2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w - 1, x2)
        y2 = min(frame_h - 1, y2)

        current_roi = (x1, y1, x2, y2)
        # 처음 클릭 시 기준 중심도 세팅
        last_cx = (x1 + x2) / 2
        last_cy = (y1 + y2) / 2
        print(f"New ROI set: {current_roi}")


def main():
    global current_roi, last_cx, last_cy

    ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)
    model = YOLO("yolo11n.pt")
    video_path = "/home/nes/Timeline1.mov"

    frame_count = 0
    prev_time = time.time()

    results_generator = model(
        source=video_path,
        stream=True,
        device=0,
        imgsz=416,
        verbose=False
    )

    first_result = next(results_generator)
    first_frame = first_result.orig_img
    frame_h, frame_w, _ = first_frame.shape

    cv2.namedWindow("YOLOv11 ROI Follow Smooth")
    mouse_param = {"frame_shape": (frame_h, frame_w)}
    cv2.setMouseCallback("YOLOv11 ROI Follow Smooth", mouse_callback, mouse_param)

    results_iter = [first_result]

    for r in results_generator:
        results_iter.append(r)

        while results_iter:
            r = results_iter.pop(0)

            frame_count += 1
            print(f"\n=== Frame {frame_count} ===")
            if ser:
                ser.write(f"=== Frame {frame_count} Start ===\n".encode())

            frame = r.orig_img.copy()
            boxes = r.boxes

            # ===== 1) ROI가 있으면, 중심에 가장 가까운 객체를 찾고, 점프 제한 + lerp 적용 =====
            if current_roi is not None and len(boxes) > 0:
                rx1, ry1, rx2, ry2 = current_roi
                # 기준 중심: 직전 타겟이 있으면 그 좌표, 없으면 ROI 중심
                if last_cx is None or last_cy is None:
                    base_cx = (rx1 + rx2) / 2
                    base_cy = (ry1 + ry2) / 2
                else:
                    base_cx = last_cx
                    base_cy = last_cy

                min_dist = 1e18
                target_cx, target_cy = None, None

                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    bx1, by1, bx2, by2 = xyxy
                    cx = (bx1 + bx2) / 2
                    cy = (by1 + by2) / 2

                    dist = (cx - base_cx) ** 2 + (cy - base_cy) ** 2

                    if dist < min_dist:
                        min_dist = dist
                        target_cx, target_cy = cx, cy

                if target_cx is not None:
                    # 너무 멀리 점프하는 경우는 무시 (ROI 유지)
                    if min_dist <= MAX_JUMP_DIST2:
                        # lerp로 부드럽게 이동
                        if last_cx is None or last_cy is None:
                            new_cx = target_cx
                            new_cy = target_cy
                        else:
                            new_cx = (1 - LERP_ALPHA) * last_cx + LERP_ALPHA * target_cx
                            new_cy = (1 - LERP_ALPHA) * last_cy + LERP_ALPHA * target_cy

                        x1 = int(new_cx - ROI_W / 2)
                        y1 = int(new_cy - ROI_H / 2)
                        x2 = int(new_cx + ROI_W / 2)
                        y2 = int(new_cy + ROI_H / 2)

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame_w - 1, x2)
                        y2 = min(frame_h - 1, y2)

                        current_roi = (x1, y1, x2, y2)
                        last_cx, last_cy = new_cx, new_cy
                    else:
                        print("Jump too large, keep ROI as is.")
            # =======================================================================

            # ROI 사각형 표시
            if current_roi is not None:
                x1, y1, x2, y2 = current_roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # ===== 2) ROI 안 박스만 처리 =====
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                bx1, by1, bx2, by2 = xyxy
                cx = (bx1 + bx2) / 2
                cy = (by1 + by2) / 2

                class_name = model.names[cls_id]

                if current_roi is None:
                    continue

                rx1, ry1, rx2, ry2 = current_roi
                inside_roi = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

                if inside_roi:
                    print(f"Class: {class_name}, Confidence: {conf:.2f}, Box: {xyxy}")
                    if ser:
                        data = f"Class: {class_name}, Confidence: {conf:.2f}, Box: {xyxy}\n"
                        ser.write(data.encode())

                    cv2.rectangle(
                        frame,
                        (int(bx1), int(by1)),
                        (int(bx2), int(by2)),
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"{class_name} {conf:.2f}",
                        (int(bx1), int(by1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

            # ===== 3) FPS =====
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            print(f"FPS: {fps:.2f}")
            if ser:
                ser.write(f"FPS: {fps:.2f}\n".encode())

            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("YOLOv11 ROI Follow Smooth", frame)

            if ser:
                ser.write(f"=== Frame {frame_count} End ===\n".encode())

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                ser.close()
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):
                current_roi = None
                last_cx, last_cy = None, None
                print("ROI reset")

    cv2.destroyAllWindows()
    if ser:
        ser.close()


if __name__ == "__main__":
    main()



