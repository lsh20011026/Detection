import cv2
import numpy as np

class GUI:
    def __init__(self, window_name: str):
        self.window_name = window_name

    def setup_mouse_callback(self, callback):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, callback)
        print(f"🖱️ 마우스 콜백 '{self.window_name}' 설정 완료")

    def render_frame(self, frame, tracker, fps: float):
        # ROI1 (KCF 영역)
        if tracker.is_tracking_valid() and tracker.roi1:
            x1, y1, w, h = tracker.roi1
            if tracker.lost_frames == 0:
                cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 3)
            else:
                cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 165, 255), 2)
                cv2.putText(frame, f"Lost: {tracker.lost_frames}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 🔥 ROI2: tracker.best_roi2_bbox 재사용 (YOLO FREE!)
        if tracker.bytetrack_active and tracker.roi1 and tracker.best_roi2_bbox:
            x1, y1, w, h = tracker.roi1
            bx1, by1, bw, bh = tracker.best_roi2_bbox
            
            fx1, fy1 = x1+bx1, y1+by1
            fx2, fy2 = fx1+bw, fy1+bh
            
            # 가장 가까운 객체만 빨간색 두꺼운 선
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
            
            # ROI1 중앙 연결선
            roi_center_x, roi_center_y = x1 + w//2, y1 + h//2
            obj_center_x, obj_center_y = fx1 + bw//2, fy1 + bh//2
            cv2.line(frame, (int(roi_center_x), int(roi_center_y)), 
                    (int(obj_center_x), int(obj_center_y)), (255, 0, 255), 2)
            
            # 라벨 (conf 생략 - tracker에서 계산)
            dist = np.sqrt((obj_center_x-roi_center_x)**2 + (obj_center_y-roi_center_y)**2)
            label = f"ROI2 D:{dist:.0f}px {bw}x{bh}"
            cv2.putText(frame, label, (fx1, fy1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # FPS / 상태
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        status = "ByteTrack ON" if tracker.bytetrack_active else "KCF Only"
        color = (0, 255, 255) if tracker.bytetrack_active else (255, 255, 0)
        cv2.putText(frame, f"Dual: {status}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if tracker.fixed_roi_size:
            rw, rh = tracker.fixed_roi_size
            cv2.putText(frame, f"Tgt:{rw}x{rh} x{tracker.config.ROI_SCALE}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.putText(frame, "Click:Track | B:Toggle | R:Reset | N:NextCam | Q:Quit", 
                   (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, frame)

    def get_key(self, delay: int = 1) -> int:
        return cv2.waitKey(delay) & 0xFF



