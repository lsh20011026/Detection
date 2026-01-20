import serial
import struct
import time
import sys
import termios
import tty
import select

class SerialLogReceiver:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.ser = None
        self.port = port
        self.baudrate = baudrate
        self.rx_buffer = bytearray()
        self.packet_count = 0
        self.running = True
        self.start_time = None
        self.old_settings = None

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01)
            print("=" * 80)
            print(f"🚁 DRONE HYBRID TRACKER - 실시간 로그")
            print(f"📡 포트: {self.port} | 속도: {self.baudrate}")
            print(f"📊 F(프레임) | MODE | ROI(x,y) | CONF | FPS | 상태")
            print(f"💡 'q' 키를 눌러 종료하세요")
            print("=" * 80)
            self._enable_raw_mode()
            return True
        except Exception as e:
            print(f"❌ 연결실패: {e}")
            return False

    def _enable_raw_mode(self):
        fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    def _disable_raw_mode(self):
        if self.old_settings:
            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSAFLUSH, self.old_settings)

    def run(self):
        while self.running:
            if not self.ser or not self.ser.is_open:
                time.sleep(0.1)
                continue

            try:
                # stdin과 serial을 함께 select로 non-blocking 체크
                ready, _, _ = select.select([sys.stdin, self.ser], [], [], 0.01)
                
                # q 키 체크
                if sys.stdin in ready:
                    key = sys.stdin.read(1)
                    if key == 'q':
                        print("\n💡 q 키 감지 - 종료 중...")
                        break
                
                # serial 데이터 체크
                if self.ser in ready:
                    data = self.ser.read(self.ser.in_waiting or 2048)
                    if data:
                        self.rx_buffer.extend(data)
                        self._parse_and_log()
                        
            except KeyboardInterrupt:
                break
            except:
                time.sleep(0.01)

        self._cleanup()

    def _parse_and_log(self):
        i = 0
        while i < len(self.rx_buffer) - 46:
            if self.rx_buffer[i:i+2] == b'\xAA\x55':
                try:
                    packet = self.rx_buffer[i:i+46]

                    frame_id = struct.unpack('<I', packet[10:14])[0]
                    x1, y1, x2, y2 = struct.unpack('<IIII', packet[14:30])
                    conf = struct.unpack('<f', packet[30:34])[0]
                    mode_id = struct.unpack('<I', packet[34:38])[0]
                    fps = struct.unpack('<f', packet[38:42])[0]

                    mode_names = ['NONE   ', 'TEMPLATE', 'YOLO   ', 'UNK    ']
                    mode = mode_names[mode_id] if mode_id < 4 else 'UNK    '

                    status_id = struct.unpack('<I', packet[42:46])[0]
                    status = ['OK', 'LOST', 'ERROR'][status_id] if status_id < 3 else 'UNK'

                    self.packet_count += 1

                    print(f"📡 F{frame_id:4d} | {mode} | "
                          f"ROI({x1:4d},{y1:4d},{x2:4d},{y2:4d}) | "
                          f"C:{conf:.3f} | FPS:{fps:.1f} | {status}")

                except:
                    pass

                i += 46
            else:
                i += 1

        self.rx_buffer = self.rx_buffer[i:]

    def _cleanup(self):
        self.running = False
        self._disable_raw_mode()
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("\n" + "=" * 80)
        if self.start_time:
            print(f"🏁 로그 종료 | 총 {self.packet_count} 패킷 수신")
            print(f"⏱️   평균 {self.packet_count/(time.time()-self.start_time):.1f} FPS")
        else:
            print("🏁 로그 종료")

def main():
    receiver = SerialLogReceiver(port='/dev/ttyUSB0')

    if receiver.connect():
        receiver.start_time = time.time()
        try:
            receiver.run()
        except KeyboardInterrupt:
            receiver._cleanup()
        except Exception as e:
            receiver._cleanup()
            print(f"예외 발생: {e}")

if __name__ == "__main__":
    main()



