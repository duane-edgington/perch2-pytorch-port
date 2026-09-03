#!/usr/bin/env python3
# =============================================================================
# Perch 2.0 benchmark on the DGX Spark (GB10)            [RUN ON THE SPARK]
# =============================================================================
# Produces the poster's benchmark table: latency (mean/median/std/min) and
# batched throughput (clips/sec) for BOTH ONNX models on BOTH providers.
#
#   models    : perch_v2 (DFT)  and  perch_v2_no_dft (DFT->MatMul)
#   providers : CPUExecutionProvider  and  CUDAExecutionProvider (GB10)
#   batches   : 1, 4, 8, 16, 32
#
# The GPU onnxruntime wheel you installed exposes BOTH providers, so this needs
# no wheel swapping. onnxruntime's run() is synchronous (results land on host),
# so wall-clock timing around it is correct for CUDA without extra syncing.
#
# Prereq: reference bundle at REFS (only one *_input.npy clip is needed as the
# base waveform). Run inside your cu130 venv.
# =============================================================================

import os, sys, glob, time, statistics as stats
import numpy as np

# ------------------------------- config --------------------------------------
REFS = os.path.expanduser("~/perch-pytorch/perch2_refs")
REPO = "justinchuby/Perch-onnx"
MODELS = [("perch_v2 (DFT)", "perch_v2.onnx"),
          ("perch_v2_no_dft", "perch_v2_no_dft.onnx")]
BATCHES = [1, 4, 8, 16, 32]
TARGET_PEAK = 0.25
WARMUP = 8
MAX_TIMED = 40          # cap iterations per cell
MAX_SECONDS = 8.0       # ...or stop after this much wall time, whichever first
MIN_TIMED = 8
OUT_MD = os.path.expanduser("~/perch-pytorch/benchmark_results.md")

os.system(f"{sys.executable} -m pip install -q onnx huggingface_hub 2>/dev/null")
# IMPORTANT: import torch BEFORE onnxruntime. The onnxruntime-gpu (sm_121) wheel
# needs libcudnn.so.9 at runtime but does not bundle it. PyTorch's cu130 wheel
# DOES bundle cuDNN 9; importing torch first loads that .so into the process so
# onnxruntime's CUDA provider can dlopen it. Without this, ORT silently falls
# back to CPU (the "libcudnn.so.9: cannot open shared object file" symptom).
import torch  # noqa: F401  (loaded for its bundled cuDNN, not used directly)
import onnxruntime as ort
from huggingface_hub import hf_hub_download

print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu'}")
print(f"onnxruntime {ort.__version__} | providers: {ort.get_available_providers()}")
HAVE_CUDA = "CUDAExecutionProvider" in ort.get_available_providers()
if not HAVE_CUDA:
    print("  !! No CUDAExecutionProvider -- GPU rows will be skipped.")
    print("  !! Install the GB10 wheel (uninstall pip onnxruntime first).")

PROVIDERS = [("CPU", ["CPUExecutionProvider"])]
if HAVE_CUDA:
    # allow CPU fallback for nodes without a CUDA kernel (e.g. DFT)
    PROVIDERS.append(("CUDA(GB10)", ["CUDAExecutionProvider", "CPUExecutionProvider"]))

# --------------------------- base input --------------------------------------
def peak_norm(x, p=TARGET_PEAK):
    m = np.max(np.abs(x))
    return (x * (p / m)).astype(np.float32) if m > 0 else x.astype(np.float32)

clips = sorted(glob.glob(os.path.join(REFS, "*_input.npy")))
assert clips, f"No *_input.npy under {REFS}"
base = peak_norm(np.load(clips[0]).astype(np.float32))   # (160000,)
print(f"Base waveform: {clips[0]}  shape={base.shape}\n")

def make_batch(b):
    return np.repeat(base[np.newaxis, :], b, axis=0).astype(np.float32)  # (b,160000)

# ------------------------------ timing ---------------------------------------
def bench_cell(sess, in_name, b):
    x = make_batch(b)
    feed = {in_name: x}
    for _ in range(WARMUP):
        sess.run(None, feed)
    times = []
    t_start = time.perf_counter()
    while len(times) < MAX_TIMED:
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000.0)   # ms
        if len(times) >= MIN_TIMED and (time.perf_counter() - t_start) > MAX_SECONDS:
            break
    mean = stats.mean(times)
    return dict(
        n=len(times),
        mean=mean,
        median=stats.median(times),
        std=stats.pstdev(times) if len(times) > 1 else 0.0,
        mn=min(times),
        thru=b / (mean / 1000.0),          # clips/sec
        per_clip=mean / b,                 # ms/clip amortized
    )

# ------------------------------ sweep ----------------------------------------
rows = []   # (model, provider, batch, metrics)
for mname, fn in MODELS:
    path = hf_hub_download(repo_id=REPO, filename=fn)
    for pname, provs in PROVIDERS:
        try:
            sess = ort.InferenceSession(path, providers=provs)
        except Exception as e:
            print(f"[skip] {mname} / {pname}: session failed: {type(e).__name__}: {str(e)[:120]}")
            continue
        active = sess.get_providers()
        # Guard: if we asked for CUDA but it silently fell back to CPU-only,
        # do NOT record misleading rows. This is usually the cuDNN load-order
        # issue (import torch before onnxruntime) or a missing GPU wheel.
        if pname.startswith("CUDA") and "CUDAExecutionProvider" not in active:
            print(f"[skip] {mname} / {pname}: CUDA provider did NOT load "
                  f"(active={active}). Likely libcudnn.so.9 not resolvable. "
                  f"These would be CPU numbers -- refusing to mislabel them.\n")
            continue
        in_name = sess.get_inputs()[0].name
        for b in BATCHES:
            try:
                m = bench_cell(sess, in_name, b)
                rows.append((mname, pname, b, m))
                print(f"{mname:16s} {pname:10s} b={b:<2d}  "
                      f"mean={m['mean']:8.2f}ms  med={m['median']:8.2f}  std={m['std']:6.2f}  "
                      f"min={m['mn']:8.2f}  thru={m['thru']:7.1f} clips/s  "
                      f"per_clip={m['per_clip']:6.2f}ms  (n={m['n']})")
            except Exception as e:
                print(f"[skip] {mname}/{pname}/b={b}: {type(e).__name__}: {str(e)[:100]}")
        print(f"   ^ active providers: {active}\n")

# ------------------------------ markdown -------------------------------------
def md_tables(rows):
    L = []
    L.append("### Latency (ms per run) — mean ± std (min)\n")
    L.append("| Model | Provider | " + " | ".join(f"b={b}" for b in BATCHES) + " |")
    L.append("|---|---|" + "|".join(["---"]*len(BATCHES)) + "|")
    for mname, fn in MODELS:
        for pname, _ in PROVIDERS:
            cells = []
            for b in BATCHES:
                r = next((m for (mm, pp, bb, m) in rows if mm==mname and pp==pname and bb==b), None)
                cells.append(f"{r['mean']:.2f} ± {r['std']:.2f} ({r['mn']:.2f})" if r else "—")
            L.append(f"| {mname} | {pname} | " + " | ".join(cells) + " |")
    L.append("\n### Throughput (clips/sec)\n")
    L.append("| Model | Provider | " + " | ".join(f"b={b}" for b in BATCHES) + " |")
    L.append("|---|---|" + "|".join(["---"]*len(BATCHES)) + "|")
    for mname, fn in MODELS:
        for pname, _ in PROVIDERS:
            cells = []
            for b in BATCHES:
                r = next((m for (mm, pp, bb, m) in rows if mm==mname and pp==pname and bb==b), None)
                cells.append(f"{r['thru']:.1f}" if r else "—")
            L.append(f"| {mname} | {pname} | " + " | ".join(cells) + " |")
    L.append("\n### Amortized latency per clip (ms/clip)\n")
    L.append("| Model | Provider | " + " | ".join(f"b={b}" for b in BATCHES) + " |")
    L.append("|---|---|" + "|".join(["---"]*len(BATCHES)) + "|")
    for mname, fn in MODELS:
        for pname, _ in PROVIDERS:
            cells = []
            for b in BATCHES:
                r = next((m for (mm, pp, bb, m) in rows if mm==mname and pp==pname and bb==b), None)
                cells.append(f"{r['per_clip']:.2f}" if r else "—")
            L.append(f"| {mname} | {pname} | " + " | ".join(cells) + " |")
    return "\n".join(L)

md = md_tables(rows)
print("\n" + "="*70 + "\nMARKDOWN (paste into the poster Results panel)\n" + "="*70)
print(md)
with open(OUT_MD, "w") as f:
    f.write(f"# Perch 2.0 GB10 benchmark\n\n"
            f"- onnxruntime {ort.__version__}, providers {ort.get_available_providers()}\n"
            f"- base clip: {os.path.basename(clips[0])}, 5 s @ 32 kHz, peak-normed 0.25\n"
            f"- warmup {WARMUP}, up to {MAX_TIMED} timed iters / {MAX_SECONDS}s per cell\n\n")
    with open(OUT_MD, "a") as f2:
        pass
with open(OUT_MD, "a") as f:
    f.write(md + "\n")
print(f"\nSaved: {OUT_MD}")
