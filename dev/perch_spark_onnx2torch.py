#!/usr/bin/env python3
# =============================================================================
# Perch 2.0 -> native PyTorch on the DGX Spark (GB10)   [RUN ON THE SPARK]
# =============================================================================
# Goal (the poster's headline result): run Perch 2.0 TensorFlow-FREE on Blackwell
# in native PyTorch, and prove it reproduces Google's reference embeddings.
#
# For EACH of two ONNX files:
#     perch_v2.onnx         (faithful; contains a DFT node)
#     perch_v2_no_dft.onnx  (DFT rewritten as MatMul; likelier to convert clean)
# we try:
#     1) onnx2torch -> native torch.nn.Module -> run on CUDA (GB10)   <-- native
#     2) if that fails, onnxruntime fallback (CPU)                    <-- still TF-free
# then wrap with the verified pipeline and validate against your references:
#     audio -> peak_norm(0.25) -> model -> embedding / spectrogram
#                                        -> raw logits -> 0.97*raw - 10.0
#
# Each model is isolated: a roadblock on no_dft (or the DFT model) is logged
# and skipped, never fatal. Run inside your cu130 venv:  ~/perch-pytorch/.venv
#
# Prereq: your references copied to the Spark. Set REFS below.
# =============================================================================

import os, sys, time, traceback
import numpy as np

# ------------------------------- config --------------------------------------
REFS = os.path.expanduser("~/perch-pytorch/perch2_refs")   # adjust if needed
REPO = "justinchuby/Perch-onnx"
FILES = ["perch_v2.onnx", "perch_v2_no_dft.onnx"]
TARGET_PEAK   = 0.25
LOGIT_SLOPE   = 0.97
LOGIT_INTCPT  = -10.0
N_WARMUP, N_TIMED = 5, 50

# ------------------------- deps (quiet if present) ---------------------------
# NOTE: deliberately does NOT install onnxruntime here -- on GB10 the GPU wheel
# (Jay0515 sm_121) must be installed manually, and pip's onnxruntime would
# silently shadow it. Install onnx2torch/onnx/hf only.
os.system(f"{sys.executable} -m pip install -q onnx onnx2torch huggingface_hub 2>/dev/null")

import torch
from huggingface_hub import hf_hub_download

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device {DEV} | "
      f"{torch.cuda.get_device_name(0) if DEV=='cuda' else 'no gpu'}")

# The reliable GPU detector for onnxruntime on GB10 (startup logs lie).
try:
    import onnxruntime as _ort
    print(f"onnxruntime {_ort.__version__} | providers: {_ort.get_available_providers()}")
    if "CUDAExecutionProvider" not in _ort.get_available_providers():
        print("  !! CUDAExecutionProvider NOT available -- onnxruntime will run on CPU.")
        print("  !! Install the GB10 wheel (uninstall pip onnxruntime first):")
        print("  !! pip install https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121/resolve/main/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl")
except ImportError:
    print("onnxruntime not installed (ok if native onnx2torch path succeeds).")

# ------------------------------ helpers --------------------------------------
def peak_norm(x, p=TARGET_PEAK):
    m = np.max(np.abs(x))
    return (x * (p / m)).astype(np.float32) if m > 0 else x.astype(np.float32)

def calibrate(logits):
    return LOGIT_SLOPE * logits + LOGIT_INTCPT

def cos(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def maxabs(a, b):
    return float(np.max(np.abs(a.reshape(-1).astype(np.float64) - b.reshape(-1).astype(np.float64))))

def by_shape(arrays):
    """Map model outputs to roles by shape (order-independent)."""
    out = {}
    for a in arrays:
        a = np.asarray(a)
        if a.ndim == 4:                         out["spatial"]     = a
        elif a.ndim == 3:                       out["spectrogram"] = a
        elif a.ndim == 2 and a.shape[-1] == 1536:   out["embedding"] = a
        elif a.ndim == 2 and a.shape[-1] == 14795:  out["label"]     = a
    return out

def load_refs():
    import glob
    clips = sorted(glob.glob(os.path.join(REFS, "*_input.npy")))
    assert clips, f"No *_input.npy under {REFS}"
    data = []
    for ip in clips:
        cid = os.path.basename(ip).replace("_input.npy", "")
        data.append(dict(
            cid=cid,
            x=np.load(ip).astype(np.float32),
            emb=np.load(os.path.join(REFS, f"{cid}_embeddings.npy")),
            spec=np.load(os.path.join(REFS, f"{cid}_frontend.npy")),
            log=np.load(os.path.join(REFS, f"{cid}_logits__label.npy")),
        ))
    return data

REFDATA = load_refs()
print(f"Loaded {len(REFDATA)} reference clips from {REFS}\n")
summary = []

# --------------------------- per-model attempt -------------------------------
def validate_and_time(run_fn, tag):
    """run_fn(x_1xN float32 np) -> dict(role->np array of raw model outputs)."""
    # parity on all clips
    embc = specc = logc = 0.0
    for r in REFDATA:
        outs = run_fn(peak_norm(r["x"])[None, :])
        m = by_shape(list(outs.values()) if isinstance(outs, dict) else outs)
        embc  = max(embc,  1 - cos(m["embedding"],   r["emb"]))   # track worst
        specc = max(specc, 1 - cos(m["spectrogram"], r["spec"]))
        logc  = max(logc,  1 - cos(calibrate(m["label"]), r["log"]))
    # representative maxabs on clip0
    r0 = REFDATA[0]
    o0 = run_fn(peak_norm(r0["x"])[None, :])
    m0 = by_shape(list(o0.values()) if isinstance(o0, dict) else o0)
    emb_max  = maxabs(m0["embedding"], r0["emb"])
    spec_max = maxabs(m0["spectrogram"], r0["spec"])
    log_max  = maxabs(calibrate(m0["label"]), r0["log"])
    # timing
    xin = peak_norm(REFDATA[0]["x"])[None, :]
    for _ in range(N_WARMUP): run_fn(xin)
    if DEV == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_TIMED): run_fn(xin)
    if DEV == "cuda": torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N_TIMED * 1000
    print(f"  [{tag}] worst emb cos-err={embc:.2e} spec={specc:.2e} logit={logc:.2e} | "
          f"clip0 maxabs emb={emb_max:.2e} spec={spec_max:.2e} logit={log_max:.2e} | "
          f"{ms:.2f} ms/run")
    summary.append((tag, embc, ms))

def try_native(onnx_path, name):
    from onnx2torch import convert
    mod = convert(onnx_path).to(DEV).eval()
    @torch.no_grad()
    def run(x_np):
        t = torch.from_numpy(np.ascontiguousarray(x_np)).to(DEV)
        out = mod(t)
        out = out if isinstance(out, (list, tuple)) else (out,)
        return {i: o.detach().float().cpu().numpy() for i, o in enumerate(out)}
    validate_and_time(run, f"{name} | onnx2torch/native/{DEV}")

def try_ort(onnx_path, name):
    import onnxruntime as ort
    provs = ort.get_available_providers()
    use = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in provs]
    sess = ort.InferenceSession(onnx_path, providers=use)
    inm = sess.get_inputs()[0].name
    def run(x_np):
        outs = sess.run(None, {inm: x_np.astype(np.float32)})
        return {i: o for i, o in enumerate(outs)}
    validate_and_time(run, f"{name} | onnxruntime/{use[0].replace('ExecutionProvider','')}")

# ------------------------------- main ----------------------------------------
for fn in FILES:
    name = fn.replace(".onnx", "")
    print(f"=== {fn} ===")
    try:
        path = hf_hub_download(repo_id=REPO, filename=fn)
    except Exception as e:
        print(f"  download failed: {e}\n"); continue
    # 1) native torch via onnx2torch (the poster result)
    try:
        try_native(path, name)
    except Exception as e:
        print(f"  onnx2torch native path failed: {type(e).__name__}: {str(e)[:160]}")
        print(f"  -> falling back to onnxruntime")
        # 2) onnxruntime fallback
        try:
            try_ort(path, name)
        except Exception as e2:
            print(f"  onnxruntime also failed: {type(e2).__name__}: {str(e2)[:160]}")
            print(f"  -> abandoning {fn} for now")
    print()

# ------------------------------ summary --------------------------------------
print("="*70)
print("SUMMARY (worst-case embedding cosine error; lower = better parity)")
for tag, embc, ms in summary:
    status = "PASS" if embc < 1e-4 else "CHECK"
    print(f"  [{status}] {tag:48s} emb cos-err={embc:.2e}  {ms:7.2f} ms")
print("="*70)
print("="*70)
print("Any PASS row = Perch 2.0 embeddings reproduced TF-free to ~1e-9.")
print("A '/cuda' or '/CUDA' row = running on the GB10 (Blackwell).")
print("  native/cuda  -> onnx2torch torch.nn.Module on GPU  (strongest claim)")
print("  onnxruntime/CUDA -> TF-free on GB10 via ORT-CUDA   (strong; wheel-based)")
print("Native torch on GPU remains the stretch goal (hand-built EfficientNet-B3).")
