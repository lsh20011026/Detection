# camera_manager.py
import cv2
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CameraInfo:
    index: int
    width: int
    height: int

class CameraManager:
    def __init__(self, config):
        self.config = config
        self.cameras: List[CameraInfo] = []
        self.current_camera: Optional[CameraInfo] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_w = 0
        self.frame_h = 0

    def detect_available_cameras(self):
        self.cameras = []
        for i in range(self.config.max_cameras):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    self.cameras.append(CameraInfo(i, w, h))
                cap.release()
        print(f"사용 가능 카메라: {[c.index for c in self.cameras]}")

    def init_camera(self, cam_index: int) -> bool:
        if self.cap:
            self.cap.release()

        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        fourcc = cv2.VideoWriter_fourcc(*self.config.fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffersize)

        if not cap.isOpened():
            print(f"카메라 열기 실패: index={cam_index}")
            self.cap = None
            return False

        self.cap = cap
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.current_camera = next(
            (c for c in self.cameras if c.index == cam_index),
            CameraInfo(cam_index, self.frame_w, self.frame_h)
        )
        print(f"카메라 오픈: index={cam_index} ({self.frame_w}x{self.frame_h})")
        return True

    def switch_to_next(self):
        if len(self.cameras) <= 1:
            print("전환할 카메라 없음")
            return False

        if self.current_camera is None:
            print("current_camera 없음 → 첫 카메라부터 시작")
            return self.init_camera(self.cameras[0].index)

        cur_idx = self.cameras.index(self.current_camera)
        next_idx = (cur_idx + 1) % len(self.cameras)
        next_cam = self.cameras[next_idx]
        print(f"카메라 전환 요청 → index={next_cam.index}")
        return self.init_camera(next_cam.index)

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def cleanup(self):
        if self.cap:
            self.cap.release()
            self.cap = None



