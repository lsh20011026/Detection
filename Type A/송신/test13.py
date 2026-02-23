import cv2
import numpy as np
import time
from ultralytics import YOLO
import serial
import sys
import struct
sys.path.append("/home/nes/.local/lib/python3.10/site-packages")

class HybridTracker:
    """YOLO + Template Matching 하이브리드 추적 시스템 (시리얼 전송 포함)"""
    
    def __init__(self):
        # 추적 상태
        self.current_roi = None
        self.template = None
        self.tracking_mode = "NONE"  # "NONE" / "TEMPLATE"
        self.yolo_enabled = False
        
        # 파라미터 (설정 가능)
        self.ROI_W = 60
        self.ROI_H = 60
        self.REDETECT_INTERVAL = 10
        self.TEMPLATE_CONF_THRESH = 0.7
        self.MAX_LOST_FRAMES = 30
        
        # 상태 변수
        self.frame_h = 0
        self.frame_w = 0
        self.lost_frame_count = 0
        self.frame_count = 0
        self.last_conf = 0.0
        
        # 시리얼 전송
        self.last_tx_frame = 0
        self.tx_interval = 5  # 5프레임마다 전송
        self.ser = None
        
        # 하드웨어
        self.model = None
        self.cap = None
        
        # 마우스 콜백용
        self.mouse_param = {"frame": None, "boxes": None}
    
    def init_hardware(self, video_path):
        """시리얼, YOLO, 비디오 초기화"""
        try:
            self.ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)
            print("✅ Serial connected")
        except:
            self.ser = None
            print("⚠️  Serial unavailable")
        
        # TensorRT 엔진 로드
        self.model = YOLO("/home/nes/yolo11n.engine", task='detect')
        print("🚀 TensorRT YOLO loaded")
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"❌ Cannot open {video_path}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📹 Video: {self.frame_w}x{self.frame_h}")
    
    def send_serial_data(self, frame_id, roi, conf, mode, fps, status):
        """시리얼 데이터 전송"""
        if self.ser is None or not self.ser.is_open:
            return
        
        try:
            # 패킷: AA55 + timestamp(8) + frame_id(4) + roi(16) + conf(4) + mode(4) + fps(4) + status(4)
            timestamp = int(time.time() * 1000)
            mode_id = {'NONE':0, 'TEMPLATE':1, 'YOLO':2}.get(mode, 0)
            status_id = {'OK':0, 'LOST':1 if self.lost_frame_count > 10 else 0, 'ERROR':2}.get(status, 2)
            
            packet = struct.pack('<Q', timestamp) + \
                     struct.pack('<I', frame_id) + \
                     struct.pack('<IIII', *map(int, roi or (0,0,0,0))) + \
                     struct.pack('<f', float(conf)) + \
                     struct.pack('<I', mode_id) + \
                     struct.pack('<f', float(fps)) + \
                     struct.pack('<I', status_id)
            
            packet = b'\xAA\x55' + packet
            self.ser.write(packet)
            
        except Exception as e:
            print(f"TX error: {e}")
    
    def setup_window(self):
        """윈도우 및 마우스 콜백 설정"""
        win_name = "HybridTracker (Drone)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)
        cv2.setMouseCallback(win_name, self.mouse_callback, self.mouse_param)
        return win_name
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 처리"""
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param["frame"]
            boxes = param["boxes"]
            
            clicked_on_object = self._handle_yolo_click(x, y, boxes, frame)
            if not clicked_on_object:
                self._handle_manual_roi(x, y, frame)
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            self._handle_zoom(flags)
    
    def _handle_yolo_click(self, x, y, boxes, frame):
        """YOLO 박스 클릭 처리"""
        if boxes is None or not self.yolo_enabled:
            return False
        
        for box in boxes:
            b_xyxy = box.xyxy[0].tolist()
            if b_xyxy[0] <= x <= b_xyxy[2] and b_xyxy[1] <= y <= b_xyxy[3]:
                self._set_roi_from_box(b_xyxy, frame, shrink=0.1)
                print(f"[YOLO→TEMPLATE] ROI: {self.current_roi}")
                self.lost_frame_count = 0
                return True
        return False
    
    def _handle_manual_roi(self, x, y, frame):
        """수동 ROI 설정"""
        x1 = max(0, int(x - self.ROI_W / 2))
        y1 = max(0, int(y - self.ROI_H / 2))
        x2 = min(self.frame_w - 1, int(x + self.ROI_W / 2))
        y2 = min(self.frame_h - 1, int(y + self.ROI_H / 2))
        
        if x2 > x1 and y2 > y1:
            self.current_roi = (x1, y1, x2, y2)
            self.template = frame[y1:y2, x1:x2].copy()
            self.tracking_mode = "TEMPLATE"
            self.lost_frame_count = 0
            print(f"[MANUAL] ROI: {self.current_roi}")
    
    def _set_roi_from_box(self, xyxy, frame, shrink=0.1):
        """박스에서 ROI 생성"""
        x1, y1, x2, y2 = map(int, xyxy)
        w, h = x2 - x1, y2 - y1
        x1 = int(x1 + w * shrink)
        x2 = int(x2 - w * shrink)
        y1 = int(y1 + h * shrink)
        y2 = int(y2 - h * shrink)
        
        self.current_roi = (max(0, x1), max(0, y1), min(self.frame_w-1, x2), min(self.frame_h-1, y2))
        self.template = frame[y1:y2, x1:x2].copy()
        self.tracking_mode = "TEMPLATE"
    
    def _handle_zoom(self, flags):
        """마우스 휠 줌"""
        win_name = "HybridTracker (Drone)"
        rect = cv2.getWindowImageRect(win_name)
        w, h = rect[2], rect[3]
        
        if flags > 0:  # 확대
            new_w, new_h = min(1920, w + 100), min(1080, h + 100)
        else:  # 축소
            new_w, new_h = max(640, w - 100), max(480, h - 100)
        
        cv2.resizeWindow(win_name, new_w, new_h)
    
    def template_matching(self, frame):
        """템플릿 매칭 추적"""
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
                
                # 로그
                new_cx = max_loc[0] + sx1
                new_cy = max_loc[1] + sy1
                drift_dist = np.sqrt((new_cx - roi_cx)**2 + (new_cy - roi_cy)**2)
                
                self._log_template(frame_count=self.frame_count, max_val=max_val, 
                                 roi=self.current_roi, new_pos=(new_cx, new_cy), 
                                 drift=drift_dist)
                
                self.last_conf = max_val
                
                if max_val > self.TEMPLATE_CONF_THRESH:
                    x1, y1 = max_loc[0] + sx1, max_loc[1] + sy1
                    x2, y2 = x1 + tw, y1 + th
                    self.current_roi = (max(0, x1), max(0, y1), 
                                      min(self.frame_w-1, x2), min(self.frame_h-1, y2))
                    self.template = frame[int(y1):int(y2), int(x1):int(x2)].copy()
                    self.lost_frame_count = 0
                    return True, max_val
                else:
                    self.lost_frame_count += 1
                    return False, max_val
            else:
                self.lost_frame_count += 1
                return False, 0.0
        except Exception as e:
            print(f"💥 Template error: {e}")
        return False, 0.0
    
    def _log_template(self, frame_count, max_val, roi, new_pos, drift):
        """템플릿 로그 출력"""
        print(f"F{frame_count:4d} | TMP:{max_val:.3f} | "
              f"ROI{roi}→NEW{new_pos} | DRIFT:{drift:.1f}px")
    
    def yolo_detection(self, frame):
        """YOLO 객체 탐지"""
        self.mouse_param["boxes"] = None
        
        if not self.yolo_enabled:
            return
        
        try:
            results = self.model.predict(
                source=frame, device=0, verbose=False, 
                conf=0.35, imgsz=320, max_det=5
            )
            
            boxes = None
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    self.mouse_param["boxes"] = boxes
                    self._draw_yolo_boxes(r, frame)
                    break
            
            self._yolo_redetect(boxes, frame)
            
        except Exception as e:
            print(f"YOLO error: {e}")
    
    def _draw_yolo_boxes(self, result, frame):
        """YOLO 박스 그리기"""
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), 
                         (int(xyxy[2]), int(xyxy[3])), (128, 128, 128), 1)
    
    def _yolo_redetect(self, boxes, frame):
        """ROI 내 YOLO 재탐지"""
        if (self.frame_count % self.REDETECT_INTERVAL != 0 or 
            self.current_roi is None or self.tracking_mode != "TEMPLATE"):
            return
        
        rx1, ry1, rx2, ry2 = self.current_roi
        roi_cx = (rx1 + rx2) / 2
        roi_cy = (ry1 + ry2) / 2
        
        best_box, best_score, best_conf = self._find_best_roi_box(boxes, roi_cx, roi_cy, rx1, rx2, ry1, ry2)
        
        if best_box is not None:
            self._set_roi_from_box(best_box, frame)
            print(f"[REDETECT✓] conf={best_conf:.3f}")
            self.lost_frame_count = 0
    
    def _find_best_roi_box(self, boxes, roi_cx, roi_cy, rx1, rx2, ry1, ry2):
        """ROI 내 최적 박스 찾기"""
        best_box = None
        best_score = -1
        best_conf = 0
        
        if boxes is None:
            return best_box, best_score, best_conf
        
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            conf = float(box.conf[0])
            
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                dist2 = (cx - roi_cx)**2 + (cy - roi_cy)**2
                score = -dist2
                
                if score > best_score:
                    best_score = score
                    best_box = xyxy
                    best_conf = conf
        
        return best_box, best_score, best_conf
    
    def draw_roi(self, frame):
        """ROI 시각화"""
        if self.current_roi is not None:
            x1, y1, x2, y2 = map(int, self.current_roi)
            color = (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, "TEMPLATE", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def draw_status(self, frame, fps):
        """상태 표시"""
        status = f"M:{self.tracking_mode[:4]} Y:{'ON' if self.yolo_enabled else 'OFF'} L:{self.lost_frame_count}"
        cv2.putText(frame, status, (10, self.frame_h - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "t:YOLO r:reset q:quit Wheel:ZOOM TX:ON", (10, self.frame_h - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS:{fps:.1f}", (10, self.frame_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def process_frame(self, frame):
        """단일 프레임 처리"""
        self.frame_count += 1
        self.mouse_param["frame"] = frame
        
        # 진행률 표시
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            progress = self.frame_count / total_frames * 100
            cv2.putText(frame, f"F:{self.frame_count}/{total_frames} ({progress:.1f}%)",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        fps = 0.0
        # 1. 템플릿 매칭
        if (self.current_roi is not None and 
            self.tracking_mode == "TEMPLATE" and self.template is not None):
            
            success, conf = self.template_matching(frame)
            self.last_conf = conf
            color = (0, 255, 255) if success else (0, 128, 255)
            text = f"TEMPLATE {conf:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            fps = 30.0  # 추정값
        
        # 2. YOLO 탐지
        self.yolo_detection(frame)
        
        # 3. ROI 그리기
        self.draw_roi(frame)
        
        # 4. 시리얼 전송 (5프레임마다)
        if self.frame_count % self.tx_interval == 0:
            status = 'LOST' if self.lost_frame_count > 10 else 'OK'
            self.send_serial_data(
                frame_id=self.frame_count,
                roi=self.current_roi,
                conf=self.last_conf,
                mode=self.tracking_mode,
                fps=fps,
                status=status
            )
        
        return frame
    
    def handle_keys(self, key, win_name):
        """키 입력 처리"""
        if key == ord('q'):
            return False
        elif key == ord('r'):
            self.reset_tracking()
            print("🔄 Reset")
        elif key == ord('t'):
            self.yolo_enabled = not self.yolo_enabled
            print(f"YOLO {'ON' if self.yolo_enabled else 'OFF'}")
        return True
    
    def reset_tracking(self):
        """추적 리셋"""
        self.current_roi = None
        self.template = None
        self.tracking_mode = "NONE"
        self.lost_frame_count = 0
    
    def run(self, video_path):
        """메인 루프"""
        self.init_hardware(video_path)
        win_name = self.setup_window()
        
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"🎬 Total: {total_frames} | t=YOLO, r=RESET, q=QUIT | 📡 Serial TX:ON")
        
        prev_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("🏁 End of video")
                break
            
            frame = self.process_frame(frame)
            
            # FPS 계산
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if self.frame_count > 1 else 0
            prev_time = curr_time
            self.draw_status(frame, fps)
            
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            
            if not self.handle_keys(key, win_name):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """정리"""
        if self.cap:
            self.cap.release()
        if self.ser:
            self.ser.close()
        cv2.destroyAllWindows()
        print("👋 Tracker ended")

if __name__ == "__main__":
    tracker = HybridTracker()
    tracker.run("/home/nes/Timeline1.mov")



