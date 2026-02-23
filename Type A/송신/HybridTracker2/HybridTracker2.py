import cv2          # 이미지 처리
import numpy as np  # 수치 계산
import time         # 시간 측정
from ultralytics import YOLO     # YOLO 객체 탐지
import serial       # 직렬 통신
import sys
import struct
sys.path.append("/home/nes/.local/lib/python3.10/site-packages")
from KalmanTracker import KalmanTracker   # 칼만 필터 추적기
from serial_manager import SerialManager  # 직렬 통신 매니저
from config import TrackerConfig          # 추적 설정
from camera_manager import CameraManager  # 카메라 관리
from ui_manager import UIManager          # UI 관리 (마우스/키보드/그리기)

class HybridTracker:
    """
    하이브리드 객체 추적기: YOLO + 템플릿 매칭 + 칼만 필터
    - 실시간 비디오에서 객체 탐지 및 추적
    - 직렬 통신으로 드론/하드웨어 전송
    """

    def __init__(self):
        # 설정 및 매니저 초기화
        self.config = TrackerConfig()
        
        self.serial_mgr = SerialManager()      # 직렬 포트 관리
        self.last_tx_frame = 0                 # 마지막 전송 프레임
        self.kalman_tracker = KalmanTracker()  # 칼만 필터
        self.camera_mgr = CameraManager(self.config)  # 카메라
        self.ui_mgr = UIManager(self, self.camera_mgr) # UI
        
        # 프레임 크기 (동적 업데이트)
        self.frame_w = 0
        self.frame_h = 0
        
        # 추적 상태 변수
        self.current_roi = None          # 현재 ROI (x1,y1,x2,y2)
        self.template = None             # 템플릿 이미지
        self.tracking_mode = "NONE"      # NONE/TEMPLATE/KALMAN_ONLY
        self.yolo_enabled = False        # YOLO 활성화 여부
        self.roi_tracking_active = False # ROI 추적 활성
        self.show_yolo_boxes = True       # YOLO 박스 표시
        self.lost_frame_count = 0        # 추적 실패 카운트
        self.frame_count = 0             # 총 프레임 수
        self.last_conf = 0.0             # 마지막 신뢰도
        self.kalman_only_count = 0       # 칼만 전용 카운트

        self.model = None                 # YOLO 모델

    def init_hardware(self):
        """카메라 및 YOLO 모델 초기화"""
        self.camera_mgr.detect_available_cameras()
        if not self.camera_mgr.cameras:
            raise ValueError("❌ 사용 가능한 카메라가 없습니다")
        
        self.camera_mgr.init_camera(self.camera_mgr.cameras[0].index)
        self.frame_w, self.frame_h = self.camera_mgr.frame_w, self.camera_mgr.frame_h
        print(f"📹 초기 카메라: {self.camera_mgr.current_camera.index} ({self.frame_w}x{self.frame_h})")
        
        try:
            self.model = YOLO(self.config.MODEL_PATH, task='detect')
            print("🚀 TensorRT YOLO loaded")
        except Exception as e:
            print(f"❌ YOLO model load failed: {e}")
            self.model = None

    def send_serial_data(self, frame_id, roi, conf, mode, fps, status):
        """추적 데이터 직렬 전송 (SerialManager 위임)"""
        if not self.serial_mgr.is_connected():
            return
        self.serial_mgr.send_tracking_data(frame_id, roi, conf, mode, fps, status)

    # ================= Kalman 필터 관련 ==================
    def _init_kalman(self, cx, cy):
        """칼만 필터 초기화 (중심점 기준)"""
        self.kalman_tracker.init_kalman(cx, cy)

    def _reset_kalman(self):
        """칼만 필터 리셋"""
        self.kalman_tracker.reset()

    def _predict_kalman_roi(self):
        """칼만 필터로 다음 ROI 예측"""
        success, roi = self.kalman_tracker.predict_roi(
            self.frame_w, self.frame_h, self.config.ROI_W, self.config.ROI_H
        )
        if success:
            self.current_roi = roi
            self.kalman_only_count += 1
            self.lost_frame_count = 0
            self.tracking_mode = "KALMAN_ONLY"
            self.kalman_tracker.use_for_tracking = True
        return success

    def _fallback_to_kalman(self):
        """템플릿 실패시 칼만 필터 폴백"""
        if self.kalman_tracker.initialized and self._predict_kalman_roi():
            print(f"🔥 KALMAN_ONLY[{self.kalman_only_count}] activated")
        else:
            self.lost_frame_count += 1

    # ================= 템플릿 매칭 ==================
    def template_matching(self, frame):
        """템플릿 매칭 + 칼만 보정"""
        if self.template is None or self.current_roi is None:
            return False, 0.0

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tpl_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            th, tw = tpl_gray.shape[:2]

            rx1, ry1, rx2, ry2 = self.current_roi
            roi_cx, roi_cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2

            margin = 80
            sx1 = max(0, rx1 - margin)
            sy1 = max(0, ry1 - margin)
            sx2 = min(self.frame_w, rx2 + margin)
            sy2 = min(self.frame_h, ry2 + margin)

            if (sx2 - sx1) > tw and (sy2 - sy1) > th:
                search_roi = gray[sy1:sy2, sx1:sx2]
                res = cv2.matchTemplate(search_roi, tpl_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                meas_x = max_loc[0] + sx1 + tw / 2.0
                meas_y = max_loc[1] + sy1 + th / 2.0
                drift_dist = np.sqrt((meas_x - roi_cx) ** 2 + (meas_y - roi_cy) ** 2)

                self._log_template(
                    frame_count=self.frame_count,
                    max_val=max_val,
                    roi=self.current_roi,
                    new_pos=(meas_x, meas_y),
                    drift=drift_dist
                )

                self.last_conf = max_val

                if max_val > self.config.TEMPLATE_CONF_THRESH:
                    x1 = int(meas_x - tw / 2)
                    y1 = int(meas_y - th / 2)
                    x2 = x1 + tw
                    y2 = y1 + th

                    self.current_roi = (max(0, x1), max(0, y1),
                                        min(self.frame_w - 1, x2), min(self.frame_h - 1, y2))
                    self.template = frame[int(y1):int(y2), int(x1):int(x2)].copy()
                    self.lost_frame_count = 0
                    self.kalman_only_count = 0
                    self.tracking_mode = "TEMPLATE"

                    if self.kalman_tracker.initialized:
                        self.kalman_tracker.correct(meas_x, meas_y)

                    self.kalman_tracker.use_for_tracking = False
                    return True, max_val
                else:
                    self.lost_frame_count += 1
                    self._fallback_to_kalman()
                    return False, max_val
            else:
                self.lost_frame_count += 1
                self._fallback_to_kalman()
                return False, 0.0

        except Exception as e:
            print(f"💥 Template error: {e}")
            self._fallback_to_kalman()
        return False, 0.0

    def _log_template(self, frame_count, max_val, roi, new_pos, drift):
        """템플릿 매칭 결과 로그 출력"""
        print(f"F{frame_count:4d} | TMP:{max_val:.3f} | "
              f"ROI{roi}→NEW{new_pos} | DRIFT:{drift:.1f}px")

    # ================= YOLO 관련 ==================
    def _set_roi_from_box(self, xyxy, frame, shrink=0.1):
        """YOLO 박스에서 ROI 및 템플릿 생성"""
        x1, y1, x2, y2 = map(int, xyxy)
        w, h = x2 - x1, y2 - y1
        x1 = int(x1 + w * shrink)
        x2 = int(x2 - w * shrink)
        y1 = int(y1 + h * shrink)
        y2 = int(y2 - h * shrink)

        self.current_roi = (max(0, x1), max(0, y1),
                            min(self.frame_w - 1, x2), min(self.frame_h - 1, y2))
        self.template = frame[y1:y2, x1:x2].copy()
        self.tracking_mode = "TEMPLATE"
        self.kalman_only_count = 0
        self.roi_tracking_active = True

        cx = (self.current_roi[0] + self.current_roi[2]) / 2
        cy = (self.current_roi[1] + self.current_roi[3]) / 2
        self._init_kalman(cx, cy)

    def yolo_detection(self, frame):
        """YOLO 객체 탐지 + 박스 그리기 + 재탐지"""
        self.ui_mgr.mouse_param["boxes"] = None

        if not self.yolo_enabled or self.model is None:
            return

        try:
            results = self.model.predict(
                source=frame, device=0, verbose=False,
                conf=self.config.YOLO_CONF, imgsz=self.config.YOLO_IMGSZ, 
                max_det=self.config.YOLO_MAX_DET
            )

            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    self.ui_mgr.mouse_param["boxes"] = boxes
                    self.ui_mgr.draw_yolo_boxes(r, frame)
                    break

            self._yolo_redetect(boxes, frame)

        except Exception as e:
            print(f"YOLO error: {e}")

    def _yolo_redetect(self, boxes, frame):
        """주기적 ROI 내 YOLO 재탐지"""
        if (self.frame_count % self.config.REDETECT_INTERVAL != 0 or
                self.current_roi is None):
            return

        rx1, ry1, rx2, ry2 = self.current_roi
        roi_cx = (rx1 + rx2) / 2
        roi_cy = (ry1 + ry2) / 2

        best_box, best_score, best_conf = self._find_best_roi_box(
            boxes, roi_cx, roi_cy, rx1, rx2, ry1, ry2
        )

        if best_box is not None:
            self._set_roi_from_box(best_box, frame)
            print(f"[REDETECT✓] conf={best_conf:.3f}")
            self.lost_frame_count = 0
            self.kalman_only_count = 0

    def _find_best_roi_box(self, boxes, roi_cx, roi_cy, rx1, rx2, ry1, ry2):
        """ROI 내 최적 YOLO 박스 선택 (신뢰도 + 거리 기준)"""
        best_box = None
        best_score = -1
        best_conf = 0

        if boxes is None or len(boxes) == 0:
            return best_box, best_score, best_conf

        for box in boxes:
            try:
                xyxy = box.xyxy[0].tolist()
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                conf = float(box.conf[0])

                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    dist2 = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2
                    score = conf * 1000 - dist2

                    if score > best_score:
                        best_score = score
                        best_box = xyxy
                        best_conf = conf
            except:
                continue

        return best_box, best_score, best_conf

    # ================= 핵심 프레임 처리 ==================
    def process_frame(self, frame):
        """
        단일 프레임 처리 (추적 로직만)
        - 템플릿/칼만 우선 처리
        - 실패시 YOLO 재탐지
        - 직렬 전송
        """
        self.frame_count += 1
        self.ui_mgr.mouse_param["frame"] = frame
        self.frame_w, self.frame_h = self.camera_mgr.frame_w, self.camera_mgr.frame_h

        tracking_success = False
        fps_est = 0.0

        if (self.current_roi is not None and self.tracking_mode in ["TEMPLATE", "KALMAN_ONLY"] 
            and self.template is not None):
            
            if self.tracking_mode == "TEMPLATE":
                success, conf = self.template_matching(frame)
                self.last_conf = conf
                tracking_success = success
                fps_est = 30.0
            else:  # KALMAN_ONLY
                tracking_success = True
                self.last_conf = 0.75
                fps_est = 60.0

        if not tracking_success:
            if self.lost_frame_count > self.config.MAX_LOST_FRAMES:
                print("💥 MAX_LOST → FULL RESET")
                self.reset_tracking()
            elif self.kalman_only_count > self.config.KALMAN_ONLY_FRAMES:
                print("💥 KALMAN_TIMEOUT → YOLO REDETECT")
                self.template = None

        self.yolo_detection(frame)

        if self.frame_count % self.config.TX_INTERVAL == 0:
            status = 'LOST' if self.lost_frame_count > 10 else 'OK'
            self.send_serial_data(
                frame_id=self.frame_count,
                roi=self.current_roi,
                conf=self.last_conf,
                mode=self.tracking_mode,
                fps=fps_est,
                status=status
            )

        return frame

    def reset_tracking(self):
        """전체 추적 상태 리셋"""
        self.current_roi = None
        self.template = None
        self.tracking_mode = "NONE"
        self.lost_frame_count = 0
        self.kalman_only_count = 0
        self.roi_tracking_active = False
        self.show_yolo_boxes = True
        self._reset_kalman()

    def cleanup(self):
        """리소스 정리 (카메라/직렬/UI)"""
        if hasattr(self, 'camera_mgr'):
            self.camera_mgr.cleanup()
        self.serial_mgr.close()
        if hasattr(self, 'ui_mgr'):
            self.ui_mgr.cleanup()
        print("👋 Tracker ended")

    def run(self):
        """메인 루프: 하드웨어 초기화 → 프레임 처리 → UI 표시"""
        self.init_hardware()
        win_name = self.ui_mgr.setup_window()

        print(f"🎬 CameraManager stream | t=YOLO b=BBOX n:NEXT_CAM r=RESET q=QUIT")
        print(f"🔥 현재 카메라: {self.camera_mgr.current_camera.index}")
        print(f"🔥 사용 가능 카메라: {[c.index for c in self.camera_mgr.cameras]}")
        print(f"🔥 YOLO REDETECT:ON | BBOX:클릭후OFF | 'n'로 카메라전환 | 📡 Serial TX:ON")

        self.ui_mgr.prev_time = cv2.getTickCount()
        while True:
            ret, frame = self.camera_mgr.read_frame()
            if not ret:
                print("💥 카메라 읽기 실패")
                time.sleep(0.1)
                continue

            frame = self.process_frame(frame)
            
            frame, fps = self.ui_mgr.prepare_display_frame(frame)
            self.ui_mgr.show_frame(frame)

            key = cv2.waitKey(1) & 0xFF
            if not self.ui_mgr.handle_keys(key):
                break

        self.cleanup()

if __name__ == "__main__":
    tracker = HybridTracker()
    tracker.run()


