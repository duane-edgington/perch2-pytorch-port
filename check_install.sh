cd ~/perch-pytorch && source venv/bin/activate
python - <<'EOF'
import torch, torchaudio, librosa, soundfile, timm, numpy
def line(k,v): print(f"{k:<24} {v}")
line("torch", torch.__version__)
line("torch.version.cuda", torch.version.cuda)
line("cuda.is_available", torch.cuda.is_available())
line("get_arch_list", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    line("device_name", torch.cuda.get_device_name(0))
    line("device_capability", torch.cuda.get_device_capability(0))
    x = torch.randn(2048, 2048, device="cuda")
    (x @ x).sum().item(); torch.cuda.synchronize()
    line("gpu_matmul_smoke_test", "PASSED")
# exercise torchaudio's mel path too, since that's what the frontend uses
mel = torchaudio.transforms.MelSpectrogram(sample_rate=32000, n_fft=640, hop_length=320, n_mels=128)
line("torchaudio_mel_shape", tuple(mel(torch.randn(1, 32000)).shape))
line("torchaudio", torchaudio.__version__)
line("librosa", librosa.__version__)
line("timm", timm.__version__)
print("\nAll imports + GPU + mel path OK." if torch.cuda.is_available() else "\nCUDA NOT available — stop and check.")
EOF
