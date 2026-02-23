#!/bin/bash
set -e

echo "=== Jetson ML: 목표 버전 정확 맞춤 ==="
echo "목표: Ultralytics 8.4.11 | OpenCV 4.5.4 | NumPy 1.23.5 | PyTorch 2.9.1 | CUDA 12.6"

# 1. 기반 + PyTorch/CUDSS
echo "1. Python + PyTorch 2.9.1..."
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo add-apt-repository ppa:saiarcot895/chromium-beta -y
sudo apt update && sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip chromium-browser
python3.10 -m pip install --upgrade pip --user

cd ~/Downloads/
pip3 install --user torch-2.9.1-cp310-cp310-linux_aarch64.whl torchvision-0.24.1-cp310-cp310-linux_aarch64.whl
wget -q https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb || true
sudo dpkg -i cudss-local-tegra-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb 2>/dev/null || true
sudo cp /var/cudss-local-tegra-repo-ubuntu2404-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/ 2>/dev/null || true
sudo apt update && sudo apt install -y cudss 2>/dev/null || true
pip3 install --user --no-deps onnx onnxsim
cd -

# 2. Ultralytics 8.4.11 먼저
echo "2. Ultralytics 8.4.11..."
pip3 uninstall -y ultralytics numpy opencv* || true
pip3 install --user --no-deps ultralytics==8.4.11

# 3. MAX 모드
sudo nvpmodel -m 0 && sudo jetson_clocks

# 4. NumPy + OpenCV 완전 제거 후 재설치
echo "3. NumPy 1.23.5 + OpenCV (APT 버전)..."
pip3 uninstall -y numpy opencv* || true
pip3 install --user --force-reinstall --no-cache-dir "numpy==1.23.5"

# 🔥 OpenCV 완전 제거 후 APT로 재설치
echo "4. OpenCV 완전 제거 및 재설치..."
pip uninstall opencv-python -y 2>/dev/null || true
pip3 uninstall -y opencv* || true
pip3 cache purge
rm -rf ~/.local/lib/python3.10/site-packages/cv2* ~/.local/lib/python3.10/site-packages/opencv*

# 패키지 업데이트
sudo apt update

# APT 버전으로 재설치
sudo apt install -y python3-opencv libopencv-dev libgtk-3-dev pkg-config

# 5. 검증 (정확 형식)
echo "5. 검증..."
python3 -c "
import ultralytics, cv2, numpy as np, torch
print(f'Ultralytics: {ultralytics.__version__}')
print(f'  OpenCV: {cv2.__version__}')
print(f'  NumPy:  {np.__version__}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:   {torch.version.cuda}')
print(f'  GPU:    {torch.cuda.is_available()}')
"

# 6. Export 스크립트
cat << EOF > ~/export_yolo.py
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='onnx')
model.export(format='engine', half=True, imgsz=256, device=0)
print('TensorRT 엔진 생성 완료!')
EOF

sudo apt autoremove -y
echo "🎉 목표 달성! python3 ~/export_yolo.py"


