"""HybridTracker 설정 관리"""
from dataclasses import dataclass

@dataclass
class TrackerConfig:
    ROI_W: int = 60
    ROI_H: int = 60
    REDETECT_INTERVAL: int = 10
    TEMPLATE_CONF_THRESH: float = 0.65
    MAX_LOST_FRAMES: int = 45
    KALMAN_ONLY_FRAMES: int = 15
    TX_INTERVAL: int = 5
    YOLO_CONF: float = 0.35
    YOLO_IMGSZ: int = 256
    YOLO_MAX_DET: int = 5
    CAM_WIDTH: int = 640
    CAM_HEIGHT: int = 480
    MODEL_PATH: str = "/home/nes/yolo11n.engine"



