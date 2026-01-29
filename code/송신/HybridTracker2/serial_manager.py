import serial
import time
import struct

class SerialManager:
    """드론 추적기용 시리얼 통신 매니저 (Jetson THS1)"""
    
    def __init__(self, port='/dev/ttyTHS1', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.connect()
    
    def connect(self):
        """Jetson UART 포트 연결 (timeout=1초)"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"✅ Serial connected: {self.port}")
            return True
        except Exception as e:
            print(f"⚠️ Serial unavailable: {e}")
            self.ser = None
            return False
    
    def send_tracking_data(self, frame_id, roi, conf, mode, fps, status):
        """추적 데이터 바이너리 패킷 전송 (AA55 헤더)"""
        if self.ser is None or not self.ser.is_open:
            return False
        
        try:
            timestamp = int(time.time() * 1000)
            mode_id = {'NONE': 0, 'TEMPLATE': 1, 'KALMAN_ONLY': 2, 'YOLO': 3}.get(mode, 0)
            status_id = {'OK': 0, 'LOST': 1, 'ERROR': 2}.get(status, 2)
            
            packet = struct.pack('<Q', timestamp) + \
                     struct.pack('<I', frame_id) + \
                     struct.pack('<IIII', *map(int, roi or (0, 0, 0, 0))) + \
                     struct.pack('<f', float(conf)) + \
                     struct.pack('<I', mode_id) + \
                     struct.pack('<f', float(fps)) + \
                     struct.pack('<I', status_id)
            
            packet = b'\xAA\x55' + packet
            self.ser.write(packet)
            return True
            
        except Exception as e:
            print(f"TX error: {e}")
            return False
    
    def close(self):
        """시리얼 포트 안전 종료"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("🔌 Serial disconnected")
    
    def is_connected(self):
        """연결 상태 확인"""
        return self.ser is not None and self.ser.is_open


