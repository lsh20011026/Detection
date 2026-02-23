from dataclasses import dataclass

@dataclass
class TrackerConfig:
    """트래커 설정값"""
    ROI_W: int = 60                    # 추적 영역 너비
    ROI_H: int = 60                    # 추적 영역 높이
    REDETECT_INTERVAL: int = 10        # 재탐지 간격(프레임)
    TEMPLATE_CONF_THRESH: float = 0.65 # 템플릿 매칭 신뢰도 임계값
    MAX_LOST_FRAMES: int = 45          # 최대 손실 프레임 수
    KALMAN_ONLY_FRAMES: int = 15       # 칼만 필터 단독 사용 프레임 수
    TX_INTERVAL: int = 5               # 전송 간격(프레임)
    YOLO_CONF: float = 0.35            # YOLO 탐지 신뢰도 임계값
    YOLO_IMGSZ: int = 256              # YOLO 입력 이미지 크기
    YOLO_MAX_DET: int = 5              # YOLO 최대 탐지 객체 수
    CAM_WIDTH: int = 640               # 카메라 너비
    CAM_HEIGHT: int = 480              # 카메라 높이
    MODEL_PATH: str = "/home/nes/yolo11n.engine"  # YOLO 모델 경로


