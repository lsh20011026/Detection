#!/bin/bash
set -e

echo "Jetson Orin Nano 환경 설정 시작..."

# 1. Chromium
sudo add-apt-repository ppa:saiarcot895/chromium-beta -y
sudo apt update && sudo apt install -y chromium-browser

# 2. Python 3.10
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update && sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip

# 3. ML 패키지
pip3 install --user ultralytics==8.4.11 "numpy==1.23.5"

# 4. OpenCV
sudo apt install -y python3-opencv libopencv-dev libgtk-3-dev pkg-config

# 5. PyTorch (Downloads에 whl 파일 필요)
cd ~/Downloads/
pip3 install torch-2.9.1-cp310-cp310-linux_aarch64.whl --user || echo "torch whl 파일 필요"
pip3 install torchvision-0.24.1-cp310-cp310-linux_aarch64.whl --user || echo "torchvision whl 파일 필요"
cd ~

# 6. CUDSS (libcudss.so 에러 해결)
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2404-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt update && sudo apt install -y cudss
rm cudss-local-tegra-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb

# 7. 한글 입력기
sudo apt install -y ibus-hangul
ibus restart
echo "한글 설정: ibus-setup 실행 후 한국어 추가, 로그아웃/로그인"

echo "완료! python3.10 -c 'import torch,cv2,ultralytics; print(\"OK\")'"

