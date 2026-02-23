# config.py
from dataclasses import dataclass

@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    buffersize: int = 1
    max_cameras: int = 4
    fourcc: str = "YUYV"

@dataclass
class TrackingConfig:
    KEEP_FRAMES: int = 30
    ROI_SCALE: float = 1.5
    BBOX_HISTORY_LEN: int = 5
    DIST_THRESHOLD: float = 100
    CONF_THRESHOLD: float = 0.3
    HIGH_CONF_THRESHOLD: float = 0.7
    roi_adjust_interval: int = 20

@dataclass
class LoggerConfig:
    show_fps: bool = True
    show_status: bool = True
    show_roi_info: bool = True


