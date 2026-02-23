import cv2
import numpy as np
import time
from ultralytics import YOLO

class CSRTTracker:
    def __init__(self, model_path='/home/nes/yolo11n.engine'):
        self.model = YOLO(model_path, task='detect')
        print("🚀 TensorRT YOLO + CSRT 로드 완료")
        self.tracker = None
        self.selected_bbox = None
        self.tracking_active = False
        self.yolo_active = True
        self.frame = None
        self.frame_w = 0
        self.frame_h = 0
        self.current_detections = []
        self.prev_detections = []   
        self.frame_count = 0
        self.last_status = ""  
        self.fps_display = "FPS: --"

    def mouse_callback(self, event, x, y, flags, param):
        """YOLO bbox 클릭 + 임의 위치 클릭 모두 지원"""
        if event == cv2.EVENT_LBUTTONDOWN and self.yolo_active:
            clicked_bbox = self._find_clicked_bbox(x, y)
            
            if clicked_bbox is None:
                w, h = 80, 80
                clicked_bbox = (max(0, x-w//2), max(0, y-h//2), w, h)
                print(f"🎯 임의 위치 클릭 → 새 bbox: {clicked_bbox}")
            else:
                print(f"✅ YOLO bbox 클릭: {clicked_bbox}")
            
            self.selected_bbox = clicked_bbox
            self._init_csrt(clicked_bbox)
            self.tracking_active = True
            self.yolo_active = False
            self.last_status = "CSRT"

    def _find_clicked_bbox(self, x, y):
        if self.current_detections:
            for bbox in self.current_detections:
                x1, y1, x2, y2 = map(int, bbox)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    w, h = 80, 80
                    return (max(0, cx-w//2), max(0, cy-h//2), w, h)
        return None

    def _init_csrt(self, bbox):
        self.tracker = cv2.TrackerCSRT_create()
        ok = self.tracker.init(self.frame, bbox)
        print("CSRT init:", "✅ OK" if ok else "❌ FAIL")

    def _find_best_detection(self):
        if not self.current_detections:
            return None
        
        # 현재 bbox가 없으면 가장 큰 detection 선택
        areas = [((x2-x1)*(y2-y1), (x1,y1,x2-x1,y2-y1)) 
                for x1,y1,x2,y2 in self.current_detections]
        if areas:
            _, best_bbox = max(areas)
            return best_bbox
        return None

    def process_frame(self, frame):
        self.frame_count += 1
        self.frame = frame.copy()
        self.frame_h, self.frame_w = frame.shape[:2]
        display = frame.copy()
        
        # 🔥 이전 detection 저장
        self.prev_detections = self.current_detections.copy()
        
        # 🔥 YOLO: 2프레임 스킵 + 이전 detection 병합
        if self.yolo_active and self.frame_count % 2 == 0:
            input_size = (256, 256)
            small_frame = cv2.resize(frame, input_size)
            results = self.model(small_frame, verbose=False, imgsz=input_size[0])
            
            scale_x, scale_y = self.frame_w / input_size[0], self.frame_h / input_size[1]
            self.current_detections = []
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box * [scale_x, scale_y, scale_x, scale_y]
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        self.current_detections.append([x1, y1, x2, y2])
            
            # 🔥 현재 detection 없으면 이전 사용
            if not self.current_detections and self.prev_detections:
                self.current_detections = self.prev_detections[:3]
        
        # 🔥 YOLO 스킵 프레임: 이전 detection 희미하게 표시
        elif self.yolo_active and self.prev_detections:
            for bbox in self.prev_detections[:3]:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 1)

        elif self.tracking_active:
            # 🔥 CSRT 트래킹
            csrt_ok, csrt_bbox = self.tracker.update(frame)
            
            if csrt_ok:
                # 🔥 CSRT bbox (파랑, 굵음)
                p1, p2 = (int(csrt_bbox[0]), int(csrt_bbox[1])), (int(csrt_bbox[0]+csrt_bbox[2]), int(csrt_bbox[1]+csrt_bbox[3]))
                cv2.rectangle(display, p1, p2, (255, 0, 0), 3)
                new_status = "CSRT OK"
                
                # 🔥 신뢰도 표시 (CSRT는 내부적으로 계산하지 않으므로 confidence=1로 표시)
                cv2.putText(display, f"CSRT:100%", (p1[0], p1[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
            else:
                # 🔥 CSRT 실패시 YOLO로 재초기화 시도
                best_det = self._find_best_detection()
                if best_det:
                    self._init_csrt(best_det)
                    p1 = (int(best_det[0]), int(best_det[1]))
                    p2 = (int(best_det[0]+best_det[2]), int(best_det[1]+best_det[3]))
                    cv2.rectangle(display, p1, p2, (0, 255, 255), 3)
                    new_status = "CSRT Re-init"
                else:
                    new_status = "CSRT FAIL → YOLO"
                    # 🔥 10프레임 연속 실패시 YOLO로 완전 전환
                    if self.frame_count % 10 == 0:
                        print("🔄 CSRT 연속 실패 → YOLO 모드 전환")
                        self.yolo_active = True
                        self.tracking_active = False
                        self.tracker = None
                        new_status = "YOLO"
            
            # 🔥 상태 변경 시에만 업데이트
            if new_status != self.last_status:
                self.last_status = new_status
            
            # 상태 표시
            color = (0, 255, 255) if "Re-init" in self.last_status else (255, 255, 255)
            cv2.putText(display, self.last_status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 🔥 고정 텍스트 (깜빡임 방지)
        mode = "CSRT" if self.tracking_active else "YOLO"
        cv2.putText(display, f"MODE: {mode}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 🔥 클릭 안내 (YOLO 모드일 때만)
        if self.yolo_active:
            cv2.putText(display, "CLICK BBOX or ANYWHERE!", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 클릭 예시 bbox
            ex_x, ex_y = self.frame_w//2, self.frame_h//2
            ex_w, ex_h = 80, 80
            cv2.rectangle(display, (ex_x-ex_w//2, ex_y-ex_h//2), 
                         (ex_x+ex_w//2, ex_y+ex_h//2), (255, 0, 255), 1)
        
        return display   

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎬 CSRT 트래커 시작! (CSRT🔵)")
        
        win_name = "YOLO-CSRT Tracker"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)
        cv2.setMouseCallback(win_name, self.mouse_callback)
        
        fps_times = []
        fps_update_counter = 0  

        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret: 
                print("프레임 끝")
                break
            
            display = self.process_frame(frame)
            
            # 🔥 FPS: 10프레임마다 업데이트
            fps_times.append(time.time())
            fps_times = fps_times[-10:]  
            fps_update_counter += 1
            
            if len(fps_times) > 1 and fps_update_counter % 10 == 0:
                cur_fps = len(fps_times) / (fps_times[-1] - fps_times[0])
                self.fps_display = f"FPS: {cur_fps:.1f}"
            elif len(fps_times) == 1:
                self.fps_display = "FPS: warming..."
            
            cv2.putText(display, self.fps_display, (10, self.frame_h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            cv2.imshow(win_name, display)
            
            elapsed = time.time() - frame_start
            sleep_ms = max(1, int(1000 * elapsed * 0.8))
            key = cv2.waitKey(sleep_ms) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('r') or key == ord('R'):
                self.yolo_active = True
                self.tracking_active = False
                self.tracker = None
                self.last_status = ""
                print("🔄 완전 리셋")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 CSRT 트래커 종료")

if __name__ == "__main__":
    tracker = CSRTTracker()
    tracker.run("/home/nes/cctv.mp4")



