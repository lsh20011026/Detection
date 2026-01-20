import cv2
import time
from ultralytics import YOLO
import serial
import sys
sys.path.append("/home/nes/.local/lib/python3.10/site-packages")


def main():
    ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)
    # ser = None  # 시리얼 없이 테스트하려면 None

    model = YOLO("yolo11n.pt")
    video_path = "/home/nes/Timeline1.mov"

    frame_count = 0
    prev_time = time.time()

    results_generator = model.track(
        source=video_path,
        tracker='bytetrack.yaml',  # 트래커 명시
        stream=True,
        persist=True,
        device=0,
        imgsz=416,
        verbose=False
    )

    for r in results_generator:
        frame_count += 1
        print(f"\n=== Frame {frame_count} ===")
        if ser:
            ser.write(f"=== Frame {frame_count} Start ===\n".encode())

        annotated_frame = r.plot()
        boxes = r.boxes
        ids = boxes.id  # tensor([...]) 또는 None

        print("raw ids:", ids)

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            if ids is not None:
                obj_id = int(ids[i])
            else:
                obj_id = -1

            class_name = model.names[cls_id]

            print(f"ID: {obj_id}, Class: {class_name}, Confidence: {conf:.2f}, Box: {xyxy}")

            if ser:
                data = f"ID: {obj_id}, Class: {class_name}, Confidence: {conf:.2f}, Box: {xyxy}\n"
                ser.write(data.encode())

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        print(f"FPS: {fps:.2f}")
        if ser:
            ser.write(f"FPS: {fps:.2f}\n".encode())

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.imshow("YOLOv11 Tracking", annotated_frame)

        if ser:
            ser.write(f"=== Frame {frame_count} End ===\n".encode())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    if ser:
        ser.close()


if __name__ == "__main__":
    main()



