#!/usr/bin/env python3
# =============================================================================
# Native Perch 2.0 on the GB10 — validate + benchmark   (RUN ON THE SPARK)
# =============================================================================
# Runs the fully native PyTorch model (perch_frontend_torch.py + perch_embedder_
# torch.py) on the GB10 GPU: confirms parity vs references, then benchmarks
# eager and torch.compile across batch sizes. This is the "native beats the
# bridge" measurement — pure PyTorch, no onnxruntime.
#
# Layout expected on the Spark:
#   this_dir/
#     perch_frontend_torch.py
#     perch_embedder_torch.py
#     perch_bench_native_spark.py   (this file)
#     perch_weights/{weights.npz, graph_manifest.json}
#     perch2_refs/{clipNN_input.npy, clipNN_embeddings.npy, ...}   (for parity)
#     const__pad1_output_0.npy       (optional: exact mel; else HTK reconstruction)
# Adjust the paths below if yours differ.
# =============================================================================
import os, time, glob, statistics as st
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "perch_weights")
REFS = os.path.join(HERE, "perch2_refs")
MEL = os.path.join(HERE, "const__pad1_output_0.npy")
BATCHES = [1, 4, 8, 16, 32]
WARMUP, TIMED = 10, 50

from perch_embedder_torch import PerchModel

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device {dev} | "
      f"{torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'}")

model = PerchModel(WEIGHTS, exact_mel_npy=MEL if os.path.exists(MEL) else None).eval().to(dev)
# frontend runs in float64 internally for the FFT; keep model params float32 on GPU.

def cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ---- parity check on real clips ---------------------------------------------
print("\n=== parity (raw audio -> embedding) vs TF reference, on GB10 ===")
with torch.no_grad():
    for ip in sorted(glob.glob(os.path.join(REFS, "clip*_input.npy"))):
        cid = os.path.basename(ip).replace("_input.npy", "")
        ep = os.path.join(REFS, f"{cid}_embeddings.npy")
        if not os.path.exists(ep):
            continue
        x = torch.from_numpy(np.load(ip)).to(dev)
        r = np.load(ep).reshape(-1)
        e = model(x).float().cpu().numpy().reshape(-1)
        print(f"  {cid}: cos={cos(e, r):.7f}  rel_err={np.linalg.norm(e-r)/np.linalg.norm(r):.3e}")

# ---- benchmark helper -------------------------------------------------------
def bench(fn, x):
    for _ in range(WARMUP):
        fn(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        fn(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return st.mean(ts), st.pstdev(ts)

base = np.load(sorted(glob.glob(os.path.join(REFS, "clip*_input.npy")))[0]).astype(np.float32)

@torch.no_grad()
def run_eager(x):
    return model(x)

variants = [("eager", run_eager)]
try:
    compiled = torch.compile(model, mode="max-autotune")
    @torch.no_grad()
    def run_compiled(x):
        return compiled(x)
    variants.append(("compile", run_compiled))
except Exception as e:
    print(f"\n(torch.compile unavailable: {type(e).__name__}: {str(e)[:80]})")

print("\n=== GB10 native PyTorch benchmark (ms/run, throughput clips/s) ===")
print(f"{'variant':9s} {'batch':>5s} {'mean_ms':>9s} {'std':>7s} {'clips/s':>9s} {'ms/clip':>8s}")
for name, fn in variants:
    for b in BATCHES:
        x = torch.from_numpy(np.repeat(base[None, :], b, 0)).to(dev)
        try:
            m, s = bench(fn, x)
            print(f"{name:9s} {b:5d} {m:9.2f} {s:7.2f} {b/(m/1000):9.1f} {m/b:8.2f}")
        except Exception as e:
            print(f"{name:9s} {b:5d}  failed: {type(e).__name__}: {str(e)[:70]}")

print("\nCompare to the ONNX-bridge numbers (no_dft GB10): ~211 clips/s @ b16.")
print("If native (esp. torch.compile) meets or beats that, it's the 'native")
print("beats the bridge' result for the poster.")
