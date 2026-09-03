#cd ~/perch-pytorch
python3 -m venv venv
source venv/bin/activate
python --version              # expect 3.12.3
which python                  # expect ~/perch-pytorch/venv/bin/python

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip3 install numpy soundfile librosa timm matplotlib
pip3 install perch-hoplite        # do NOT add [tf] or [jax] extras

# 1. Remove the CPU onnxruntime that's currently shadowing everything
pip3 uninstall -y onnxruntime onnxruntime-gpu

# 2. Install the GB10/sm_121/cp312 GPU wheel
pip3 install https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121/resolve/main/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl

# 3. Verify — this must list CUDAExecutionProvider (ignore any startup log noise)
python3 -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"

