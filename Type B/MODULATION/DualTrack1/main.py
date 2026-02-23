# main.py (완전 버전 - 즉시 실행 가능)
import cv2
import numpy as np
from ultralytics import YOLO
from config import CameraConfig, TrackingConfig
from camera_manager import CameraManager
from tracker import DualKCFByteTrack
from gui import GUI  # gui.py가 있다면

class MouseHandler:
    """마우스 클릭으로 트래커 초기화 - 버그 없는 버전"""
    def __init__(self, tracker):
        self.tracker = tracker
        self.current_frame = None

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_frame is not None:
            print(f"클릭: ({x}, {y}) - Dual Tracker 시작")
            half = 100
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            w = min(self.current_frame.shape[1] - x1, 2 * half)
            h = min(self.current_frame.shape[0] - y1, 2 * half)
            
            self.tracker.init_from_click(self.current_frame, x, y)
            print(f"초기화 영역: {x1},{y1},{w},{h}")

def main():
    camera_config = CameraConfig(width=640, height=480)
    tracking_config = TrackingConfig()

    model = YOLO("/home/nes/yolo11n.engine", task="detect")
    camera_manager = CameraManager(camera_config)
    
    print("=== 카메라 탐지 ===")
    camera_manager.detect_available_cameras()

    if not camera_manager.cameras:
        print("❌ 사용 가능한 카메라가 없습니다.")
        return

    print(f"📹 첫 번째 카메라({camera_manager.cameras[0].index}) 시작")
    if not camera_manager.init_camera(camera_manager.cameras[0].index):
        return

    tracker = DualKCFByteTrack(model, tracking_config)
    gui = GUI("Dual KCF + ByteTrack v1.5 ✅")
    
    # ✅ 마우스 핸들러 (버그 해결)
    mouse_handler = MouseHandler(tracker)
    gui.setup_mouse_callback(mouse_handler.callback)

    print("🎮 Dual KCF + ByteTrack (Dynamic ROI1 v1.5)")
    print("👆 마우스 클릭: 추적 시작 | B:ByteTrack 토글 | R:리셋 | N:다음카메라 | Q:종료")

    fps_start = cv2.getTickCount()
    frame_buffer = None

    while camera_manager.cap and camera_manager.cap.isOpened():
        ret, frame = camera_manager.read_frame()
        if not ret or frame is None:
            print("프레임 읽기 실패")
            break
        
        frame_buffer = frame.copy()
        mouse_handler.current_frame = frame_buffer  # 마우스용 프레임 업데이트

        # FPS 계산
        fps_end = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (fps_end - fps_start)
        fps_start = fps_end

        # 트래커 업데이트
        target_found, roi = tracker.update(frame)

        # 추적 실패 체크
        if tracker.is_tracking_valid() and tracker.lost_frames > tracking_config.KEEP_FRAMES:
            tracker.reset_state()
            print("⏹️  추적 실패 → IDLE 상태")

        # GUI 렌더링
        gui.render_frame(frame, tracker, fps)

        # 키 입력
        key = gui.get_key(1)
        if key == ord('q') or key == 27:  # ESC도 추가
            break
        elif key == ord('r') or key == ord('R'):
            tracker.reset_state()
            print("🔄 리셋 완료")
        elif key == ord('b') or key == ord('B'):
            tracker.toggle_bytetrack()
        elif key == ord('n') or key == ord('N'):
            tracker.reset_state()
            if camera_manager.switch_to_next():
                print("📹 카메라 전환 성공")
            else:
                print("❌ 카메라 전환 실패")

    camera_manager.cleanup()
    cv2.destroyAllWindows()
    print("✅ 프로그램 종료")

if __name__ == "__main__":
    main()


