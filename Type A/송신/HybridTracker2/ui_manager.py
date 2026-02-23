import cv2
import numpy as np

class UIManager:
    """HybridTracker UI 완전 캡슐화 (마우스/키보드/시각화)"""
    
    def __init__(self, tracker, camera_mgr):
        self.tracker = tracker
        self.camera_mgr = camera_mgr
        
        self.win_name = None
        self.window_width = 1280
        self.window_height = 720
        
        self.mouse_param = {"frame": None, "boxes": None}
        
        self.prev_time = 0
        self.fps = 0.0

    def setup_window(self):
        """OpenCV 윈도우 + 마우스 콜백 설정"""
        self.win_name = "HybridTracker (Drone) - YOLO Redetect ON + CAM_SWITCH"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, self.window_width, self.window_height)
        cv2.setMouseCallback(self.win_name, self.mouse_callback, self.mouse_param)
        return self.win_name

    def mouse_callback(self, event, x, y, flags, param):
        """좌클릭(YOLO/ROI), 휠(줌) 처리"""
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param["frame"]
            boxes = param["boxes"]

            clicked_on_object = self._handle_yolo_click(x, y, boxes, frame)
            if not clicked_on_object:
                self._handle_manual_roi(x, y, frame)

        elif event == cv2.EVENT_MOUSEWHEEL:
            self._handle_zoom(flags)

    def _handle_yolo_click(self, x, y, boxes, frame):
        """YOLO 박스 클릭 → 템플릿 추적 전환"""
        if boxes is None or len(boxes) == 0 or not self.tracker.yolo_enabled:
            return False

        for box in boxes:
            try:
                b_xyxy = box.xyxy[0].tolist()
                if (b_xyxy[0] <= x <= b_xyxy[2] and 
                    b_xyxy[1] <= y <= b_xyxy[3]):
                    self.tracker._set_roi_from_box(b_xyxy, frame, shrink=0.1)
                    print(f"[YOLO→TEMPLATE] ROI: {self.tracker.current_roi}")
                    self.tracker.lost_frame_count = 0
                    self.tracker.roi_tracking_active = True
                    self.tracker.show_yolo_boxes = False
                    self.mouse_param["boxes"] = None
                    return True
            except:
                continue
        return False

    def _handle_manual_roi(self, x, y, frame):
        """마우스 클릭 중심 수동 ROI + 템플릿/칼만 초기화"""
        x1 = max(0, int(x - self.tracker.config.ROI_W / 2))
        y1 = max(0, int(y - self.tracker.config.ROI_H / 2))
        x2 = min(self.tracker.frame_w - 1, int(x + self.tracker.config.ROI_W / 2))
        y2 = min(self.tracker.frame_h - 1, int(y + self.tracker.config.ROI_H / 2))

        if x2 > x1 and y2 > y1:
            self.tracker.current_roi = (x1, y1, x2, y2)
            self.tracker.template = frame[y1:y2, x1:x2].copy()
            self.tracker.tracking_mode = "TEMPLATE"
            self.tracker.lost_frame_count = 0
            self.tracker.kalman_only_count = 0
            self.tracker.roi_tracking_active = True
            self.tracker.show_yolo_boxes = False
            print(f"[MANUAL] ROI: {self.tracker.current_roi}")

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            self.tracker._init_kalman(cx, cy)

    def _handle_zoom(self, flags):
        """마우스 휠로 윈도우 줌 (640x480 ~ 1920x1080)"""
        rect = cv2.getWindowImageRect(self.win_name)
        w, h = rect[2], rect[3]

        if flags > 0:
            new_w, new_h = min(1920, w + 100), min(1080, h + 100)
        else:
            new_w, new_h = max(640, w - 100), max(480, h - 100)

        cv2.resizeWindow(self.win_name, new_w, new_h)

    def draw_yolo_boxes(self, result, frame):
        """YOLO 박스 그리기 (회색, show_yolo_boxes=True일 때)"""
        if not self.tracker.show_yolo_boxes:
            return
        
        for box in result.boxes:
            try:
                xyxy = box.xyxy[0].tolist()
                cv2.rectangle(frame,
                              (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])),
                              (128, 128, 128), 1)
            except:
                continue

    def draw_roi(self, frame):
        """ROI 사각형 + 모드 라벨 + 칼만 위치(KF*)"""
        if self.tracker.current_roi is not None:
            x1, y1, x2, y2 = map(int, self.tracker.current_roi)
            
            if self.tracker.tracking_mode == "TEMPLATE":
                color = (0, 255, 255)    # 노랑
            elif self.tracker.tracking_mode == "KALMAN_ONLY":
                color = (255, 0, 255)    # 자홍
            else:
                color = (0, 128, 255)    # 주황

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, self.tracker.tracking_mode, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            kx, ky = self.tracker.kalman_tracker.get_position()
            if kx is not None:
                kf_color = (0, 0, 255) if self.tracker.kalman_tracker.use_for_tracking else (0, 255, 0)
                cv2.circle(frame, (int(kx), int(ky)), 
                          5 if self.tracker.kalman_tracker.use_for_tracking else 3, kf_color, -1)
                cv2.putText(frame, "KF" + ("*" if self.tracker.kalman_tracker.use_for_tracking else ""), 
                           (int(kx + 8), int(ky)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, kf_color, 1)

    def draw_status(self, frame, fps):
        """상태바: 모드/YOLO/박스/손실프레임/카메라/FPS/단축키"""
        cam_info = f"CAM{self.camera_mgr.current_camera.index}"
        bbox_status = "BBOX:OFF" if not self.tracker.show_yolo_boxes else "BBOX:ON "
        status = (f"M:{self.tracker.tracking_mode[:4]} Y:{'ON' if self.tracker.yolo_enabled else 'OFF'} "
                 f"{bbox_status}L:{self.tracker.lost_frame_count} K:{self.tracker.kalman_only_count} "
                 f"T:{'ON' if self.tracker.roi_tracking_active else 'OFF'}")
        
        cv2.putText(frame, status, (10, self.tracker.frame_h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{cam_info} ({len(self.camera_mgr.cameras)}cams)", 
                   (10, self.tracker.frame_h - 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "t:YOLO b:BBOX n:NEXT_CAM r:reset q:quit Wheel:ZOOM TX:ON", 
                   (10, self.tracker.frame_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS:{fps:.1f} CONF:{self.tracker.last_conf:.2f}", 
                   (10, self.tracker.frame_h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def handle_keys(self, key):
        """q:종료 r:리셋 t:YOLO b:BBOX n:카메라전환"""
        if key == ord('q'):
            return False
        elif key == ord('r'):
            self.tracker.reset_tracking()
            self.tracker.show_yolo_boxes = True
            print("🔄 Reset - BBOX 복원")
        elif key == ord('t'):
            self.tracker.yolo_enabled = not self.tracker.yolo_enabled
            print(f"YOLO {'ON' if self.tracker.yolo_enabled else 'OFF'}")
        elif key == ord('b'):
            self.tracker.show_yolo_boxes = not self.tracker.show_yolo_boxes
            print(f"BBOX {'ON' if self.tracker.show_yolo_boxes else 'OFF'}")
        elif key == ord('n'):
            self.camera_mgr.switch_to_next()
            self.tracker.reset_tracking()
            self.tracker.frame_w, self.tracker.frame_h = self.camera_mgr.frame_w, self.camera_mgr.frame_h
            print(f"🔄 카메라 전환: {self.camera_mgr.current_camera.index}")
        return True

    def update_fps(self):
        """getTickCount 기반 실시간 FPS"""
        curr_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_time - self.prev_time)
        self.prev_time = curr_time
        self.fps = fps
        return self.fps

    def prepare_display_frame(self, frame):
        """ROI+상태+FPS 일괄 그리기"""
        self.draw_roi(frame)
        fps = self.update_fps()
        self.draw_status(frame, fps)
        return frame, fps

    def show_frame(self, frame):
        """프레임 표시"""
        cv2.imshow(self.win_name, frame)

    def cleanup(self):
        """OpenCV 윈도우 정리"""
        cv2.destroyAllWindows()


