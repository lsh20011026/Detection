# gui.py
import cv2
import numpy as np

class GUI:
    def __init__(self, window_name: str):
        self.window_name = window_name

    def setup_mouse_callback(self, callback):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, callback)
        print(f"🖱️  마우스 콜백 '{self.window_name}' 설정 완료")

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

        # 🔥 ROI2: ROI1 중앙 가장 가까운 ByteTrack 객체만 표시!
        if tracker.bytetrack_active and tracker.roi1:
            x1, y1, w, h = tracker.roi1
            
            # ROI1 중앙점 계산
            roi_center_x, roi_center_y = x1 + w//2, y1 + h//2
            
            roi_crop = frame[y1:y1+h, x1:x1+w]
            if roi_crop.size > 0:
                try:
                    results = tracker.model.track(roi_crop, persist=True, 
                                                conf=0.25, verbose=False, tracker="bytetrack.yaml")[0]
                    if results.boxes is not None:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        confs = results.boxes.conf.cpu().numpy()
                        ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else None
                        
                        best_box = None
                        best_dist = float('inf')
                        
                        # ROI1 중앙에서 가장 가까운 객체 찾기
                        for i, box in enumerate(boxes):
                            if confs[i] > 0.25:  # 최소 conf 필터
                                bx1, by1, bx2, by2 = box
                                obj_center_x = bx1 + (bx2-bx1)/2
                                obj_center_y = by1 + (by2-by1)/2
                                
                                # ROI1 중앙까지 거리
                                dist = np.sqrt((obj_center_x - w//2)**2 + (obj_center_y - h//2)**2)
                                
                                if dist < best_dist:
                                    best_dist = dist
                                    best_box = (bx1, by1, bx2, by2, confs[i], int(ids[i]) if ids is not None else -1)
                        
                        # 가장 가까운 객체만 표시 (빨간색 두꺼운 선)
                        if best_box:
                            bx1, by1, bx2, by2, conf, obj_id = best_box
                            fx1, fy1, fx2, fy2 = x1+int(bx1), y1+int(by1), x1+int(bx2), y1+int(by2)
                            
                            # 🔥 가장 가까운 객체만 빨간색 두꺼운 선
                            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
                            
                            # 중앙점 연결선
                            cv2.line(frame, (int(roi_center_x), int(roi_center_y)), 
                                    (fx1+int((bx2-bx1)/2), fy1+int((by2-by1)/2)), 
                                    (255, 0, 255), 2)
                            
                            # 상세 라벨
                            label = f"BEST:{conf:.2f} ID:{obj_id} D:{best_dist:.0f}px"
                            cv2.putText(frame, label, (fx1, fy1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            print(f"🎯 BEST: conf={conf:.2f}, dist={best_dist:.0f}px, ID={obj_id}")
                        else:
                            cv2.putText(frame, "No close target", (x1+10, y1+30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                except Exception as e:
                    print(f"ByteTrack 오류: {e}")

        # FPS / 상태 (기존)
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


