import cv2
import time
from ultralytics import YOLO

def main():
    # 1. YOLOv11 모델 로드
    model = YOLO("yolo11n.pt")  # YOLOv11 Nano 모델 경로

    # 2. 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    frame_count = 0
    prev_time = time.time()  # FPS 계산용

    # 3. 객체 탐지 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        frame_count += 1
        print(f"\n=== Frame {frame_count} ===")

        # 4. YOLOv11 객체 탐지
        results = model.predict(frame, device=0, verbose=False)

        # 5. 결과 프레임 그리기 및 객체 정보 출력
        for r in results:
            annotated_frame = r.plot()

            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                # 클래스 이름 가져오기
                class_name = model.names[cls_id]

                print(f"Class: {class_name}, Confidence: {conf:.2f}, Box: {xyxy}")

            # FPS 계산
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # FPS 화면 표시
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.2f}",
                (10, 30),  # 좌측 상단 위치
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # 폰트 크기
                (0, 255, 0),  # 초록색
                2
            )

            cv2.imshow("YOLOv11 Detection", annotated_frame)

        # 6. 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

