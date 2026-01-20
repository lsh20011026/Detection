import serial

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # 포트명에 맞게 수정
while True:
    line = ser.readline().decode('utf-8').rstrip()
    if line:
        print(line)



