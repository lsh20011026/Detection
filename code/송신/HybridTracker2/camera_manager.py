import cv2
from config import TrackerConfig
from dataclasses import dataclass

@dataclass
class CameraInfo:
    """카메라 인덱스, 해상도 정보"""
    index: int
    width: int
    height: int

class CameraManager:
    """다중 USB 카메라 관리 및 전환 클래스"""
    
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.cameras = []
        self.current_camera = None
        self.cap = None
        self.frame_w = 0
        self.frame_h = 0
    
    def detect_available_cameras(self):
        """USB 카메라 0-3번 자동 감지"""
        self.cameras = []
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 100 and h > 100:
                    self.cameras.append(CameraInfo(i, w, h))
                cap.release()
        print(f"📹 사용 가능 카메라: {[c.index for c in self.cameras]}")
    
    def init_camera(self, cam_index):
        """지정 카메라 초기화 및 해상도 설정"""
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAM_HEIGHT)
        
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_camera = next(c for c in self.cameras if c.index == cam_index)
        print(f"📹 Camera {cam_index}: {self.frame_w}x{self.frame_h}")
    
    def switch_to_next(self):
        """현재 카메라 → 다음 카메라 순환 전환"""
        if len(self.cameras) <= 1:
            print("❌ 전환할 카메라 없음")
            return
        
        current_idx = self.cameras.index(self.current_camera)
        next_idx = (current_idx + 1) % len(self.cameras)
        self.init_camera(self.cameras[next_idx].index)
    
    def read_frame(self):
        """현재 카메라에서 프레임 읽기"""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        return ret, frame
    
    def cleanup(self):
        """카메라 리소스 해제"""
        if self.cap:
            self.cap.release()


