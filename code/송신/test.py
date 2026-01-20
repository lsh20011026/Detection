import cv2
from ultralytics import YOLO

def main():
    # 1. YOLOv11 모델 로드
    model = YOLO("yolo11n.pt")  # YOLOv11 Nano 모델 경로

    # 2. 웹캠 열기
    cap = cv2.VideoCapture(0)  # 0번 카메라 (/dev/video0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 3. 객체 탐지 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        # 4. YOLOv11으로 객체 탐지 (GPU 사용 device=0)
        results = model.predict(frame, device=0, verbose=False)

        # 5. 결과 프레임 그리기
        for r in results:
            annotated_frame = r.plot()  # 객체 박스 그리기
            cv2.imshow("YOLOv11 Detection", annotated_frame)

        # 6. 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. 종료 처리
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



