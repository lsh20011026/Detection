**객체 탐지 시스템**

**목표환경**
- jetpack 6.2.1
- Jetson Orin Nano Developer kit
- yolo11n.pt 모델 기반 yolo11n.engine 엔진 사용

**환경 구축**

**1-1. 환경구축 (SDK Manager 설치 및 리커버리 모드)**

- 우분투 22.04 환경의 PC에 SDK Manager 설치 (버전 확인 : lsb_release -a)
- https://developer.nvidia.com/sdk-manager 접속 후 .deb ubuntu 클릭하여 다운
- sudo apt install [다운로드한 파일 경로 및 이름] 입력
- sdkmanager를 터미널에 입력

Jetson 리커버리 모드
- 점퍼 선을 Jetson orin nano의 ‘수평’ 핀 중 FC_REC와 GND에 연결
- Usb - C로 PC와 Jetson을 연결



**1-2. 환경구축 (SDK Manager 사용법)**

- SDK VERSION은 6.2.1로 설정
- Target Hardware는 Jetson orin nano
- 연결이 제대로 되어있으면 알아서 설정됨
- i accept ~~~라고 적힌 부분 체크 후 다음 단계로 진행
- 다운로드 시 나오는 사용자명 및 패스워드 설정 창 작성
- Recovery mode setup은 Automatic으로 설정
- 모든 과정이 끝나면 jetson 부팅



**1-3. 환경구축 (한글 설정)**

- sudo apt update 입력
- sudo apt install -y ibus-hangul 입력
- ibus restart 입력
- ibus-setup 입력
- Input Method -> Add -> Korean -> Hangul 선택
- 재부팅



**1-4. 환경구축 (크로미움 설치)**

- sudo apt install software-properties-common
- sudo add-apt-repository ppa:deadsnakes/ppa
- sudo apt update

파이썬 설치
- sudo apt install python3.10
- 버전 확인
- python3.10 --version



**2. 각 라이브러리 버전**

- torch : 2.8.0
- torchvision : 0.23.0
- numpy : 1.26.4
- ultralytics : 8.3.221



**3. 설치 과정**

- ultralytics 인스톨
- pip3 install ultralytics
- numpy 인스톨
- pip3 install "numpy<2" --user
- opencv 인스톨
- pip3 install "opencv-python<4.12" --user
- torchvision-0.23.0-cp310-linux_aarch64.whl, torch-2.8.0-cp310-cp310-linux_aarch64.whl 다운로드 (pypi.jetson-ai-lab.io/jp6/cu126)
- cd Downloads/
- pip3 install [파일명]
- yolo11n.pt 모델 다운 
- python 실행
- from ultralytics import YOLO
- model = YOLO('yolo11n.pt') 작성 후 실행



**3-1. .pt -> .onnx -> engine 변환 과정, JETSON MAX 모드**

엔진 생성
- from ultralytics import YOLO
- model = YOLO("yolo11n.pt")
- model.export(format="engine", half=True, imgsz=256, device=0)

JETSON MAX 모드 (FPS 상승을 위해 필요)
- sudo nvpmodel -m 0 && sudo jetson_clocks



**4. 물리적 연결**

- 노트북에는 usb로 연결
- Jetson orin nano에는 아래와 같이 연결

검정색(GND) : 6번 핀 (GND)
흰색(TX) : 8번핀 (TX)
청록색(RX) : 10번 핀 (RX)



**파일 별 변경점**

**TYPE A**

**송신 (Jetson Orin nano Developer kit)**
- test.py : yolo11n.pt를 이용하여 객체 탐지
- test1.py : 프레임별 로그 추가
- test2.py : 실시간 FPS 추가
- test3.py : Yolo11n 내장 트래커 추가
- test4.py : 시리얼 통신으로 결과값 전송 추가
- test5.py : 시리얼 통신으로 받는 결과값 형식 다듬기, 트래킹 수정
- test6.py : 영상 파일에 객체 탐지 적용
- test7.py : 프레임 캡처, 추론, 시각화를 각각 별도 스레드로 분리, 해상도 축소, 5프레임마다 화면 업데이트
- test8.py : 내부 스트림 처리
- test9.py : ROI 설정
- test10.py : ROI를 설정해서 템플릿 기억 후 유지, 객체 탐지 시 BBOX 생성 후 터치 시 객체 추적
- test11.py : 기본 모델 대신 TensorRT를 통해 engine 형태로 변환 후 적용하여 프레임 상승시킴, 리팩토링
- test12.py : 시리얼 통신 보완 (rec2.py와 시리얼 통신)
- test13.py : 칼만 필터를 적용하여 객체의 움직임을 예측하여 ROI 추적을 보완
- test14.py : BBOX를 클릭할 경우 BBOX들은 모두 OFF되고, ROI만 표시됨
- test15.py : USB 카메라로 실시간 화면에 적용 가능

**수신 (PC)**
- rec.py : 시리얼 통신을 통해 데이터를 받는 코드 (test4~test11)
- rec2.py : 시리얼 통신 보완 (test12~test15)


**TYPE B**
- kcf1.py : Yolo11n.engine을 통해 객체탐지, KCF Tracker를 기반으로 추적
- kcf2.py : 칼만 필터를 적용하여 객체의 움직임을 예측하여 ROI 추적을 보완
- kcf3.py : 객체의 BBOX가 아닌 부분을 클릭해도 ROI 생성 후 추적

- csrt1.py : Yolo11n.engine을 통해 객체탐지, CSRT Tracker를 기반으로 추적
- csrt2.py : USB 카메라를 통해 찍히는 화면에 적용
- csrt3.py : 객체의 BBOX 주변의 ROI 패딩 시각화
- csrt4.py : 객체의 BBOX를 클릭하여 생긴 ROI의 사이즈를 동적으로 변동시킴
- csrt5.py : ROI 내부에만 CSRT 트래커를 적용하여 프레임 상승시킴
- csrt6.py : 최적화 및 디스플레이 상의 복잡성을 해결하기 위해 패딩 시각화 제거
- csrt7.py : 파라미터 전달 중 발생하는 오류를 해결
