# perch2-pytorch-port

A from-scratch, TensorFlow-free reimplementation of [Google Research's Perch 2.0](https://github.com/google-research/perch)
bioacoustics **embedding model** in idiomatic PyTorch — a log mel-spectrogram frontend
plus an EfficientNet-B3 embedder — validated to reproduce the reference TensorFlow model
to floating-point precision, and running on NVIDIA Grace Blackwell (GB10 / DGX Spark)
where the prebuilt TensorFlow stack is effectively unavailable.

This is a **port**, not a new model: the architecture and trained weights are Google
Research's work (Apache-2.0). The contribution here is the faithful PyTorch reimplementation,
the recovery of a few undocumented details needed for exactness, and the end-to-end
TensorFlow-free pipeline on Blackwell. See **Attribution** and `NOTICE`.

> Status: research/preview. The embedding model is complete and validated. The
> classification head is intentionally not ported (see *Scope*).

---

## Why

Perch 2.0 classifies ~15,000 species and produces general-purpose audio embeddings used
across conservation work. It ships as TensorFlow. On newest-generation NVIDIA hardware
such as the GB10 (compute capability `sm_121`, CUDA 13, aarch64), prebuilt TensorFlow is
hard to come by while PyTorch runs natively — so the framework dependency became a
practical barrier. This repo removes it: Perch 2.0 embeddings in pure PyTorch, faster on
Blackwell than the ONNX route, with the numerical parity to prove it's the same model.

The driving application is marine soundscape monitoring at MBARI: using Perch's embeddings
(via transfer — Perch is trained on ~no marine audio) inside the perch-hoplite
agile-modeling loop to detect and classify orca and dolphin calls, humpback song, and
vessel/ROV noise on Monterey Bay MARS hydrophone recordings. The detection science and
results are reported separately (IEEE OCEANS 2026, Edgington & Ryan, MBARI —
[perch-hoplite-orcas-MBNMS](https://github.com/duane-edgington/perch-hoplite-orcas-MBNMS));
this repo is the engineering that lets that pipeline run entirely in PyTorch, no TensorFlow,
on the GB10. (Perch's frontend covers 60 Hz–16 kHz; low-frequency baleen calls like
fin/blue whale sit below its band and are handled by separate detectors.)

---

## Results (validated against the reference TensorFlow outputs)

**Numerical parity** — cosine similarity / relative error of the embedding vs the TF reference:

| Path | Device | cosine | rel. error |
|---|---|---|---|
| Native embedder (fed reference frontend) | CPU | 0.9999999 | ~7e-7 |
| Native full pipeline (raw audio → embedding) | CPU | 1.0000000 | ~1–5e-5 |
| Native full pipeline | GB10 GPU | 0.9999997 | ~8e-4 |
| ONNX bridge (cross-check, `justinchuby/Perch-onnx`) | CPU | — | ~1e-9 |

Relative error is reported alongside cosine deliberately: cosine can look ~1.0 while the
embedding magnitude is still off, so **relative L2 error is the load-bearing number**.
Verified against the **live TF model** (not just archived references), the port reproduces
TF embeddings at **relative L2 ~4×10⁻⁴ (GB10 GPU, both models same-device), cosine
≥0.9999998** across test clips — magnitude-faithful, not merely angle-aligned. (On CPU,
against archived references, the embedder path is tighter still, ~1e-5.) 
The frontend alone reproduces the reference spectrogram to ~1e-4 (synthetic
signals) and ~1e-3 (quiet real recordings; near-floor log sensitivity). GPU parity is
looser than CPU purely because of float32 / tensor-core accumulation — cosine is unchanged
for any practical embedding use (search, classification, transfer).

**Throughput on the GB10** (clips/sec, 5 s clips @ 32 kHz):

| Path | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|
| ONNX bridge (ORT-CUDA, no_dft) | 102 | 200 | 207 | 211 | 206 |
| Native PyTorch (eager) | 177 | 245 | 227 | 209 | 196 |
| **Native PyTorch (`torch.compile`)** | **350** | **635** | **607** | **533** | **514** |

Native eager matches the ONNX bridge; `torch.compile` is ~2.5× the bridge at batch 4–16
and ~5× at batch 1. Native wins because the whole graph stays on-device — the bridge's
in-graph DFT forces host↔device copies — and compile fuses the conv/SE/activation chain.
(Measured on a GB10 DGX Spark.)

---

## Related work — and what's different here

This is not the only PyTorch route to Perch 2.0, and prior efforts are worth crediting:

- **[`justinchuby/Perch-onnx`](https://huggingface.co/justinchuby/Perch-onnx)** — an ONNX
  export of Google's TF Perch 2.0. Used here as a validation oracle and as the source the
  weight-extraction script reads. Prior art, credited as such.
- **[`bghani/perchv2-pytorch`](https://github.com/bghani/perchv2-pytorch)** — a PyTorch
  route that converts the ONNX graph directly via `onnx2torch`, reaching cosine 1.0 and
  differentiability. (An earlier hand-built `timm` reconstruction in that project plateaued
  around cosine 0.97 with a large relative-L2 gap; the author documents the switch to
  onnx2torch — a useful cautionary tale about trusting cosine alone.)

**What's different in this repo:** it is a **hand-written, idiomatic `torch.nn.Module`
reimplementation** — readable PyTorch you can teach from, modify, and extend — rather than
a mechanically converted graph. It is **magnitude-faithful** (relative L2 ~1e-5 on CPU, not
just high cosine). And it carries two things specific to field deployment: the **per-window
amplitude-normalization finding** for quiet audio (see below), and a full **TensorFlow-free
perch-hoplite pipeline on Grace Blackwell**. The `dev/` folder documents how the model was
reverse-engineered, including the `onnx2torch` attempt that failed in this environment and
motivated the hand build.

If you want a drop-in differentiable graph, `bghani/perchv2-pytorch` (onnx2torch) may suit
you. If you want readable, hand-built PyTorch you can reason about and extend, this repo is
for you.

---

## What's in here

**Core (the deliverable — native model + integration):**

| File | Purpose |
|---|---|
| `perch_frontend_torch.py` | `PerchFrontend` — native log-mel frontend (`nn.Module`) |
| `perch_embedder_torch.py` | `PerchEmbedder` (EfficientNet-B3) + `PerchModel` (full raw-audio→embedding pipeline) |
| `extract_weights.py` | Extract backbone weights from the model graph → `weights.npz` + `graph_manifest.json`. Runs anywhere: CPU only, no GPU, no TensorFlow. |
| `perch_hoplite_torch_adapter.py` | Wrap the model in perch-hoplite's `EmbeddingModel`; embed an audio folder → hoplite DB (no TF). Includes the mandatory per-window peak-normalization. |
| `perch_bench_native_spark.py` | Validate + benchmark the native model on the GPU (eager vs `torch.compile`) |
| `check_adapter.py` | Direct parity check: adapter `embed()` vs reference embeddings |
| `check_db.py` | Read embeddings back out of a hoplite DB and compare to references by filename |

**Docs:** `perch2_logmel_settings.md` (verified frontend spec — the exact log-mel
parameters, validated against live TF) and `benchmark_results.md`.

**Environment / data prep (shell):** `clean_install.sh`, `check_install.sh`,
`new_32k_resample_sox.sh`.

**`dev/` — how it was reverse-engineered** (not needed to run the port; kept as the
"why and how" story). See `dev/README.md`.

---

## Getting the weights (regenerate — not distributed here)

The trained weights are Google Research's Perch 2.0 parameters. **They are not committed to
this repo.** You generate them yourself, locally, from Google's published model:

1. Obtain the Perch 2.0 ONNX export
   ([`justinchuby/Perch-onnx`](https://huggingface.co/justinchuby/Perch-onnx), a faithful
   ONNX conversion of the Kaggle-hosted Perch 2.0 TF model).
2. Run `extract_weights.py` against it to write `perch_weights/weights.npz`
   (~45 MB, backbone only) and `graph_manifest.json` locally.
3. Point `PerchModel(...)` at that directory.

This keeps the repo free of redistributed model parameters while remaining fully
reproducible. To regenerate the reference parity data, run Perch 2.0 in a TF-capable
environment (e.g. Colab, via `perch-hoplite`'s TF model) on your own clips.

**Optional exact mel filterbank.** `PerchModel(..., exact_mel_npy=...)` can load the exact
mel matrix extracted from the model graph for full fidelity, but this is **not required** —
the built-in HTK reconstruction already agrees with it to <2e-5. That array is Google-derived
and, like the weights, is regenerated locally rather than shipped here.

---

## Setup / environment

The GB10 (aarch64 / `sm_121` / CUDA 13) needs a specific CUDA-13 PyTorch build; there is no
plain `requirements.txt`. Two helper scripts are included:

```bash
git clone https://github.com/duane-edgington/perch2-pytorch-port.git
cd perch2-pytorch-port
./clean_install.sh        # creates ./venv and installs the full stack
source venv/bin/activate
./check_install.sh        # sanity-check torch/CUDA/onnxruntime
```

Gotchas learned the hard way:
- **The GB10 is `sm_121`, but stock CUDA-13 PyTorch runs on it via forward-compatibility.**
  PyTorch needed no custom build; only the optional ONNX-GPU cross-check path needed a
  source-built `sm_121` onnxruntime wheel.
- **`torch.compile` needs the Python dev headers:** `sudo apt-get install -y python3.12-dev
  build-essential`, or Inductor fails with `Python.h: No such file`. Eager runs fine without.
- **Import `torch` before `onnxruntime`** (ONNX path only) so torch's bundled cuDNN 9
  satisfies the CUDA provider; otherwise it silently falls back to CPU.
- The native PyTorch path (frontend + embedder + hoplite) needs **none** of the ONNX pieces.

---

## Quickstart

```python
import numpy as np, torch
from perch_embedder_torch import PerchModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

model = PerchModel("perch_weights").eval().to(device)           # dir you generated above; built-in HTK mel agrees with the graph to <2e-5
model = torch.compile(model)   # ~2.5x throughput; needs Python dev headers. Optional.

audio = torch.from_numpy(np.load("clip.npy")).to(device)        # (160000,) or (B,160000), 32 kHz mono, 5 s
with torch.no_grad():
    emb = model(audio)                                          # (B, 1536)
```

The frontend runs its FFT in float64 for precision; the CNN runs in float32.

---

## Use with perch-hoplite (embedding and search, no TensorFlow)

perch-hoplite's embedding database and similarity search are TensorFlow-free, so you can
build a database with this model and query it with hoplite's tools. **Its classifier is not:**
`perch_hoplite/agile/classifier.py` imports TensorFlow at module level, as do
`zoo/models_tf.py`, `zoo/taxonomy_model_tf.py`, `zoo/model_configs.py`, and `zoo/kaggle_hub.py`.
Training a linear probe through upstream hoplite therefore still pulls in Keras. For a
TF-free classifier, the companion repo
[`perch-hoplite-orcas-MBNMS`](https://github.com/duane-edgington/perch-hoplite-orcas-MBNMS)
substitutes a PyTorch linear probe (`pipeline/src/train.py`) for hoplite's Keras one and mocks
the TF import. Build a database as follows:

```bash
pip install perch-hoplite            # core only; no [tf]/[jax] extras

python perch_hoplite_torch_adapter.py \
    --audio_dir /path/to/wavs --glob '*.wav' \
    --db_dir ./hoplite_db --weights_dir ./perch_weights \
    --device cuda \
    --hop_size_s 5.0                 # 5.0 = non-overlapping; e.g. 2.5 to overlap
```

**Per-window peak normalization is applied automatically and is required.** The adapter
peak-normalizes every 5 s window to 0.25 before the model. This is idempotent with Perch's
internal peak-norm, so embeddings are unchanged for healthy-amplitude audio — but for quiet
recordings (e.g. deep-water hydrophones near peak 0.002) it is essential: without it the
port's frontend arithmetic diverges from the reference at low amplitude. See
`perch2_logmel_settings.md`.

Then point hoplite's agile search / active-learning / classifier tools at `./hoplite_db`.
The adapter does **not** call hoplite's `normalize_audio` (which subtracts the mean; Perch
does not). When joining embeddings back to source, match on the recording `filename`.

---

## The architecture (as recovered from the model)

```
audio (B,160000)
  → peak_norm(0.25)
  → frame 500×640 (hop 320, 160-sample front pad) → symmetric Hann
  → rfft(1024) → magnitude / window.sum()
  → HTK mel (513→128, 60–16 kHz, DC bin zeroed)
  → 0.1 · log(max(mel, 1e-5))                         # LOG scaling (not PCEN)
  = spectrogram (B,500,128)
  → EfficientNet-B3: stem 3×3 s2 VALID → 40ch
                     → 26 MBConv blocks (folded BatchNorm; Swish; sigmoid-gated SE)
                     → head 1×1 → 1536ch
  → global average pool
  = embedding (B,1536)
```

Two details reverse-engineered rather than documented, and essential for exactness:

- **Perch 2.0's frontend is a LOG mel-spectrogram, not PCEN.** Perch 1.0 / SurfPerch (and
  some generic docs, including Google's own repo README) describe PCEN; Perch 2.0 uses
  `0.1·log(max(mel,1e-5))`. The reference floor is exactly `0.1·ln(1e-5) = −1.15129`, and
  the model graph shows a direct `Max → Log → ×0.1` path (no temporal EMA). Confirmed
  against the live TF model's spectrogram output.
- **The stem uses VALID padding** (→ 249×63), while every other k>1 conv uses JAX `SAME`.
  Recovered from the graph's squeeze-excitation pooling divisors (15687 = 249×63). Getting
  this wrong drops embedding cosine to ~0.82.

---

## Scope and honest notes

- **Embeddings only.** The classification head is not ported; Perch's main use is its
  embeddings. Calibrated class logits, if needed, are left to the ONNX bridge.
- **The ONNX bridge is prior art**, used as a validation oracle and weight source, not
  claimed as this project's work.
- **GB10 GPU parity (~1e-3 / cos 0.9999997) is a float32 precision artifact**, not an
  error; CPU reaches cos 1.0 / rel-err ~1e-5. Reported separately.

---

## Attribution

- **Perch 2.0 model, architecture, and weights:** Google Research (Apache-2.0). Weights are
  regenerated locally via the extraction script, **not redistributed here**.
- **ONNX export:** [`justinchuby/Perch-onnx`](https://huggingface.co/justinchuby/Perch-onnx).
- **Labels / taxonomy:** cgeorgiaw/Perch (iNaturalist taxonomy).
- **EfficientNet architecture:** Tan & Le, 2019.
- **This reimplementation:** Duane R. Edgington, MBARI. Code Apache-2.0 (see `LICENSE`,
  `NOTICE`).

## Citation

Accompanies the PyTorch Conference North America 2026 poster, "Perch 2.0 and Perch-Hoplite
in Pure PyTorch: An End-to-End TensorFlow-Free Bioacoustics Pipeline" (D. R. Edgington,
MBARI). `[link/DOI to follow]`

## License

Apache-2.0 (see `LICENSE` and `NOTICE`). The Perch 2.0 model is Google Research's work under
its own Apache-2.0 release; this repo redistributes no Perch weights.
