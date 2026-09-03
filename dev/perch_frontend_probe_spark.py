#!/usr/bin/env python3
# =============================================================================
# Frontend ground-truth probe  (RUN ON THE SPARK)
# =============================================================================
# The Perch frontend is a PCEN mel-spectrogram (confirmed from Google's source).
# We know the exact formulas but need to pin the trained scalar params against
# CLEAN, high-dynamic-range data (the real marine clips are near-silent, so their
# frontend is nearly flat and can't validate the pipeline).
#
# This runs strong synthetic signals through the VALIDATED ONNX (our oracle) and
# saves each (peak-normed input, frontend output) pair, plus a dump of the ONNX
# frontend constants (mel matrix, PCEN exponents). Send the saved folder back.
#
# Output: ~/perch-pytorch/frontend_probe/{signals.npz, frontend_consts.txt, *.npy}
# =============================================================================
import os, sys, json
import numpy as np

os.system(f"{sys.executable} -m pip install -q onnx huggingface_hub 2>/dev/null")
import torch  # noqa: F401  (load bundled cuDNN before onnxruntime)
import onnxruntime as ort
import onnx
from huggingface_hub import hf_hub_download

OUT = os.path.expanduser("~/perch-pytorch/frontend_probe")
os.makedirs(OUT, exist_ok=True)
SR, N = 32000, 160000
TARGET_PEAK = 0.25

def peak_norm(x, p=TARGET_PEAK):
    m = np.max(np.abs(x))
    return (x * (p / m)).astype(np.float32) if m > 0 else x.astype(np.float32)

# ---- controlled signals with real dynamic range -----------------------------
t = np.arange(N) / SR
sigs = {}
sigs["tone_1k"]      = np.sin(2*np.pi*1000*t).astype(np.float32)
sigs["tone_5k"]      = np.sin(2*np.pi*5000*t).astype(np.float32)
sigs["chirp_60_16k"] = np.sin(2*np.pi*(60 + (16000-60)*t/t[-1]/2)*t).astype(np.float32)
sigs["white_noise"]  = np.random.default_rng(0).standard_normal(N).astype(np.float32)
# a burst: silence -> tone -> silence (tests PCEN temporal AGC clearly)
burst = np.zeros(N, np.float32); burst[N//3:2*N//3] = np.sin(2*np.pi*2000*t[N//3:2*N//3])
sigs["burst_2k"]     = burst
# two-tone + amplitude ramp (rich structure)
sigs["ramp_2tone"]   = ((0.2+0.8*t/t[-1])*(np.sin(2*np.pi*800*t)+0.5*np.sin(2*np.pi*4000*t))).astype(np.float32)

# ---- run through the validated ONNX (oracle) --------------------------------
onnx_path = hf_hub_download(repo_id="justinchuby/Perch-onnx", filename="perch_v2.onnx")
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]
def spec_of(outs):
    for n, a in zip(out_names, outs):
        a = np.asarray(a)
        if a.ndim == 3 and a.shape[-2:] == (500, 128):
            return a.reshape(500, 128)
    return None

saved = {}
for name, sig in sigs.items():
    xin = peak_norm(sig)
    outs = sess.run(None, {in_name: xin[np.newaxis, :]})
    spec = spec_of(outs)
    np.save(os.path.join(OUT, f"{name}_input.npy"), xin)
    np.save(os.path.join(OUT, f"{name}_frontend.npy"), spec)
    saved[name] = dict(in_peak=float(np.max(np.abs(xin))),
                       spec_min=float(spec.min()), spec_max=float(spec.max()),
                       spec_mean=float(spec.mean()), spec_std=float(spec.std()))
    print(f"{name:14s} spec range[{spec.min():.4f},{spec.max():.4f}] "
          f"mean={spec.mean():.4f} std={spec.std():.4f}")

np.savez(os.path.join(OUT, "signals.npz"),
         **{f"{k}_input": np.load(os.path.join(OUT, f'{k}_input.npy')) for k in sigs},
         **{f"{k}_frontend": np.load(os.path.join(OUT, f'{k}_frontend.npy')) for k in sigs})
json.dump(saved, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

# ---- dump ONNX frontend constants -------------------------------------------
# The mel matrix and PCEN exponents are constant initializers near the graph
# input. Save small/2-D ones that look like window (640,) or mel ((*,128)).
m = onnx.load(onnx_path)
lines = []
lines.append("=== node op sequence (first 60) ===")
for i, node in enumerate(m.graph.node[:60]):
    lines.append(f"{i:3d} {node.op_type:22s} in={list(node.input)[:3]} out={list(node.output)[:2]}")
lines.append("\n=== candidate constant initializers (frontend-relevant shapes) ===")
from onnx import numpy_helper
for init in m.graph.initializer:
    arr = numpy_helper.to_array(init)
    shp = arr.shape
    interesting = (arr.ndim <= 2 and (128 in shp or 640 in shp or 321 in shp
                   or 513 in shp or arr.size <= 8))
    if interesting:
        lines.append(f"{init.name:40s} shape={shp} dtype={arr.dtype} "
                     f"vals[:4]={np.asarray(arr).reshape(-1)[:4]}")
        # save the mel-matrix-like and small scalar consts for exact reuse
        if (arr.ndim == 2 and 128 in shp) or arr.size <= 8:
            safe = init.name.replace("/", "_").replace(":", "_")
            np.save(os.path.join(OUT, f"const__{safe}.npy"), arr)
open(os.path.join(OUT, "frontend_consts.txt"), "w").write("\n".join(lines))
print(f"\nSaved probe to {OUT}")
print("Send back: signals.npz, frontend_consts.txt, summary.json, and any const__*.npy")
