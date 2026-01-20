import cv2
import time
from ultralytics import YOLO
import serial
import sys
sys.path.append("/home/nes/.local/lib/python3.10/site-packages")


def main():
    ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)
    #ser = None  # 시리얼 없이 테스트하려면 None으로 처리

    # YOLO 모델 로드
    model = YOLO("yolo11n.pt")

    # 비디오 파일 열기 (파일 경로로 변경)
    video_path = "/home/nes/Timeline1.mov"  # 예시 경로
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다.")
        return

    frame_count = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("비디오 끝에 도달했습니다.")
            break

        frame_count += 1
        print(f"\n=== Frame {frame_count} ===")

        if ser:
            # 시리얼에 프레임 시작 표시
            ser.write(f"=== Frame {frame_count} Start ===\n".encode())

        results = model.track(
            source=frame,
            persist=True,
            device=0,
            verbose=False
        )

        for r in results:
            annotated_frame = r.plot()
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                obj_id = int(box.id[0]) if box.id is not None else -1
                class_name = model.names[cls_id]

                # 콘솔 출력
                print(f"ID: {obj_id}, Class: {class_name}, Confidence: {conf:.2f}, Box: {xyxy}")

                # 시리얼로 전송 (테스트용 생략 가능)
                if ser:
                    data = f"ID: {obj_id}, Class: {class_name}, Confidence: {conf:.2f}, Box: {xyxy}\n"
                    ser.write(data.encode())

            # FPS 계산
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

    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()


if __name__ == "__main__":
    main()



