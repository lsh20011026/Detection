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
- ppa를 다룰 유틸리티 설치 : sudo apt install software-properties-common
- 이전 파이썬 버전 설치를 위한 세팅 : sudo add-apt-repository ppa:deadsnakes/ppa
- 패키지 최신화 : sudo apt update
- 파이썬 설치 : sudo apt install python3.10
- 버전 확인 : python3.10 --version





**2. 각 라이브러리 버전**

- Ultralytics: 8.4.11
- OpenCV: 4.5.4
- NumPy:  1.23.5
- PyTorch: 2.9.1
- CUDA:   12.6
- GPU:    True

- 버전 확인
python -c "import ultralytics, cv2, numpy as np, torch; \
print(f'Ultralytics: {ultralytics.__version__}'); \
print(f'OpenCV:      {cv2.__version__}'); \
print(f'NumPy:       {np.__version__}'); \
print(f'PyTorch:     {torch.__version__}'); \
print(f'CUDA:        {torch.version.cuda}'); \
print(f'GPU:         {torch.cuda.is_available()}')"



**3. 설치 과정**

sudo apt install python3-pip									                                                  #pip3 설치
torchvision-0.24.1-cp310-linux_aarch64.whl, torch-2.9.1-cp310-cp310-linux_aarch64.whl 다운로드
(pypi.jetson-ai-lab.io/jp6/cu126)
cd Downloads/
pip3 install torchvision-0.24.1-cp310-cp310-linux_aarch64.whl			                              #torchvision 설치
pip3 install torch-2.9.1-cp310-cp310-linux_aarch64.whl				                                  #torch 설치
pip install opencv-contrib-python 								                                              # KCF 트래커 설치
pip3 install "numpy==1.23.5" --user							                                                # numpy 설치
sudo apt install python3-opencv libopencv-dev libgtk-3-dev pkg-config 	                        # opencv 설치
pip3 install ultralytics==8.4.11								                                                # ultralytics 설치


**3-1. yolo11n.pt 모델 다운 **
python 실행
from ultralytics import YOLO
model = YOLO('yolo11n.pt') 작성 후 실행




**3-2. .pt -> .onnx -> engine 변환 과정, JETSON MAX 모드**

엔진 생성
- from ultralytics import YOLO
- model = YOLO("yolo11n.pt")
- model.export(format="engine", half=True, imgsz=256, device=0)

JETSON MAX 모드 (FPS 상승을 위해 필요)
- sudo nvpmodel -m 0 && sudo jetson_clocks


**3-3. libcudss.so 파일 누락 문제 해결**
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2404-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudss

**3-4 Mousecallback 문제 발생 시**

pip 버전 제거
pip uninstall opencv-python -y
pip3 uninstall -y opencv* || true
pip3 cache purge
rm -rf ~/.local/lib/python3.10/site-packages/cv2* ~/.local/lib/python3.10/site-packages/opencv*

패키지 업데이트
sudo apt update

apt 버전으로 재설치
sudo apt install python3-opencv libopencv-dev libgtk-3-dev pkg-config




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

**MODULATION**
- HybridTracker1 : test15.py 모듈화
- HybridTracker2 : 주/야간 카메라 전환 기능 추가

**수신 (PC)**
- rec.py : 시리얼 통신을 통해 데이터를 받는 코드 (test4~test11)
- rec2.py : 시리얼 통신 보완



**TYPE B**

**송신 (Jetson Orin nano Developer kit)**

**CSRT**
- csrt1.py : Yolo11n.engine을 통해 객체탐지, CSRT Tracker를 기반으로 추적
- csrt2.py : USB 카메라를 통해 찍히는 화면에 적용
- csrt3.py : 객체의 BBOX 주변의 ROI 패딩 시각화
- csrt4.py : 객체의 BBOX를 클릭하여 생긴 ROI의 사이즈를 동적으로 변동시킴
- csrt5.py : ROI 내부에만 CSRT 트래커를 적용하여 프레임 상승시킴
- csrt6.py : 최적화 및 디스플레이 상의 복잡성을 해결하기 위해 패딩 시각화 제거
- csrt7.py : 파라미터 전달 중 발생하는 오류를 해결

**KCF**
- kcf1.py : Yolo11n.engine을 통해 객체탐지, KCF Tracker를 기반으로 추적

**BYTETRACK**
- bytetrack1.py : 바이트 트랙 적용
- bytetrack2.py : 바이트 트랙 모드 전환 구현

**DUALTRACKER (KCF + BYTETRACK)**
- dual1.py : KCF + ByteTrack 적용 (모드 전환으로 트래커 전환)
- dual2.py : ROI 크기 문제 해결 (CLAMP 문제 해결)
- dual3.py : KCF 문제 해결
- dual4.py : 적응형 ROI 적용 (ByteTrack 한정)
- dual5.py : init 및 팅김 버그 픽스
- dual6.py : KCF + ByteTrack를 한 화면에 동시에 활용
- dual7.py : ROI 2 위치 및 크기에 맞춰 ROI 1 위치 및 크기 변경 적용
- dual8.py : 주/야간 카메라 전환

**MODULATION**
- DualTrack1 : dual8.py 모듈화
- DualTrack2 : ROI 1의 위치 및 크기 변경 시 불필요한 YOLO 사용 제거
- DualTrack3 : ROI 1 내부에서 객체의 종류에 따라 우선순위 부여
- DualTrack4 : 방향키를 통해 미세 조정이 가능한 Nudge 기능 추가


**수신 (PC)**
- rec2.py : TYPE A의 rec2.py 같은 코드
