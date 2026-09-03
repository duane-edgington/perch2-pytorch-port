#!/usr/bin/env python3
# =============================================================================
# Perch 2.0  SavedModel -> ONNX  CONVERSION + PARITY PROBE   (RUN IN COLAB)
# =============================================================================
# Decisive de-risking step. It:
#   1. Prints the TaxonomyModelTF wrapper config (normalize_audio, target_peak,
#      logit_slope/intercept, window/hop, model_path).
#   2. Converts the perch_v2 SavedModel to ONNX via tf2onnx (opset 17).
#   3. Runs the ONNX with onnxruntime (CPU) on your reference *_input.npy clips,
#      feeding BOTH raw and peak-normalized audio, and compares the ONNX
#      embedding + spectrogram to your saved TF references.
#
# What we learn:
#   - Does the SavedModel convert to ONNX at all under TF 2.20?  (the risk)
#   - Which input (raw vs peak-normalized) matches the references -> tells us
#     whether peak-normalization lives in the Python wrapper (expected) and must
#     be replicated on the Spark side.
#   - Is the ONNX numerically faithful to TF?  (cosine + max-abs-err)
#
# Prereqs in this Colab session:
#   - perch-hoplite + TF 2.20 already installed (generator cells)
#   - your reference bundle unzipped at /content/perch2_refs  (has *_input.npy,
#     *_embeddings.npy, *_frontend.npy).  Adjust REFS if needed.
# =============================================================================

import os, glob, subprocess, sys
import numpy as np

REFS      = "/content/perch2_refs"
ONNX_PATH = "/content/perch2.onnx"

# ---- 1. Wrapper config ------------------------------------------------------
from perch_hoplite.zoo import model_configs
model = model_configs.load_model_by_name('perch_v2')

cfg_keys = ["sample_rate", "window_size_s", "hop_size_s", "normalize_audio",
            "target_peak", "logit_slope", "logit_intercept", "model_path",
            "tfhub_path", "tfhub_version"]
print("="*70); print("WRAPPER CONFIG"); print("="*70)
cfg = {}
for k in cfg_keys:
    v = getattr(model, k, "<absent>")
    cfg[k] = v
    print(f"  {k:18s} = {v}")

saved_model_dir = getattr(model, "model_path", None)
print("\nSavedModel dir:", saved_model_dir)
assert saved_model_dir and os.path.isdir(saved_model_dir), \
    "Could not find the SavedModel directory via model.model_path; inspect manually."

# ---- 2. Convert SavedModel -> ONNX -----------------------------------------
print("\n" + "="*70); print("CONVERTING TO ONNX (tf2onnx)"); print("="*70)
!pip install -q tf2onnx onnx onnxruntime
# Use the CLI converter against the on-disk SavedModel dir.
cmd = [sys.executable, "-m", "tf2onnx.convert",
       "--saved-model", saved_model_dir,
       "--output", ONNX_PATH,
       "--opset", "17"]
print("running:", " ".join(cmd))
proc = subprocess.run(cmd, capture_output=True, text=True)
print(proc.stdout[-3000:])
if proc.returncode != 0:
    print("STDERR (tail):\n", proc.stderr[-4000:])
    raise RuntimeError("tf2onnx conversion failed — paste the stderr above; "
                       "we'll pivot to subgraph/weight extraction.")
print("ONNX written to", ONNX_PATH, "| size MB:",
      round(os.path.getsize(ONNX_PATH)/1e6, 1))

# ---- 3. Parity probe with onnxruntime --------------------------------------
import onnxruntime as ort
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]
print("\nONNX input:", in_name, "| outputs:", out_names)

def cos(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def maxabs(a, b):
    return float(np.max(np.abs(a.reshape(-1) - b.reshape(-1))))

def peak_norm(x, target_peak):
    p = np.max(np.abs(x))
    return (x * (target_peak / p)).astype(np.float32) if p > 0 else x

# match ONNX output names to our reference field names
def pick(outs, key):
    for n, arr in outs.items():
        if key in n.lower():
            return arr
    return None

target_peak = cfg.get("target_peak", 0.25)
clips = sorted(glob.glob(os.path.join(REFS, "*_input.npy")))
print("\n" + "="*70); print("PARITY vs REFERENCES"); print("="*70)
print(f"(target_peak = {target_peak})\n")

for ip in clips:
    cid = os.path.basename(ip).replace("_input.npy", "")
    x   = np.load(ip).astype(np.float32)
    ref_emb = np.load(os.path.join(REFS, f"{cid}_embeddings.npy"))
    ref_spec= np.load(os.path.join(REFS, f"{cid}_frontend.npy"))

    for label, audio in [("raw", x), ("peaknorm", peak_norm(x, target_peak))]:
        outs = sess.run(None, {in_name: audio[np.newaxis, :]})
        outs = dict(zip(out_names, outs))
        emb  = pick(outs, "embedding")   # may match 'embedding' or 'spatial'
        # prefer the pooled 1536 vector, not the spatial one
        emb_pooled = None
        for n, arr in outs.items():
            if "embedding" in n.lower() and "spatial" not in n.lower():
                emb_pooled = arr
        spec = pick(outs, "spectrogram")
        line = f"  {cid} [{label:8s}]"
        if emb_pooled is not None:
            line += f"  emb cos={cos(emb_pooled, ref_emb):.6f} maxabs={maxabs(emb_pooled, ref_emb):.4e}"
        if spec is not None:
            line += f" | spec cos={cos(spec, ref_spec):.6f} maxabs={maxabs(spec, ref_spec):.4e}"
        print(line)
    print()

print("Interpretation:")
print("  - The input variant (raw vs peaknorm) with cos≈1.0 is the one the")
print("    references were made with -> that's the normalization we replicate.")
print("  - emb cos≈1.0 and small maxabs => ONNX is a faithful TF-free graph.")
