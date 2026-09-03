# CLAUDE.md — context for AI assistants working in this repo

Orientation for an AI assistant (Claude Code or similar). Read before editing. Prefer these
established facts over re-deriving them — several were recovered the hard way and are easy
to get wrong.

## What this is

`perch2-pytorch-port`: a from-scratch, TensorFlow-free PyTorch reimplementation of **Google
Research's Perch 2.0** bioacoustics **embedding model** (log-mel frontend + EfficientNet-B3),
plus a perch-hoplite adapter so the whole agile-modeling workflow runs in pure PyTorch. Built
for the NVIDIA GB10 DGX Spark (aarch64, CUDA 13, `sm_121`) where prebuilt TensorFlow is
effectively unavailable.

This is a **port**, credited as such — architecture and weights are Google's (Apache-2.0).
The contribution is the reimplementation, the recovered undocumented details, and the
TF-free Blackwell pipeline. Author: Duane R. Edgington, MBARI. License: Apache-2.0.

Companion repo (the science / detection results): `duane-edgington/perch-hoplite-orcas-MBNMS`
(IEEE OCEANS 2026). This repo is the engineering that repo depends on for embeddings.

## Scope — do not add these

- **Embeddings only.** The classification head is intentionally NOT ported.
- **No classifiers.** Orca/marine classifiers live in the OCEANS repo, not here.
- **No weights or Google-derived data committed** (Option 1). `weights.npz`,
  `graph_manifest.json`, `perch2_refs/`, `const__pad1_output_0.npy`, audio — all
  regenerated locally by the user, never committed. `.gitignore` blocks them; keep it that way.

## Established facts — do NOT re-derive or "correct" these

- **Perch 2.0's frontend is LOG-mel, NOT PCEN.** Confirmed three ways: the Perch 2.0 paper
  (van Merriënboer et al., arXiv:2508.04665), reverse-engineering, and the live TF model's
  spectrogram bottoming out at exactly `0.1·ln(1e-5) = −1.151293`. Perch 1.0 / SurfPerch used
  PCEN; 2.0 does not. Google's own `google-research/perch` README says "PCEN" — that
  describes 1.0 and is the source of widespread confusion. If you see PCEN asserted for 2.0,
  it's wrong.
- **Frontend recipe:** peak_norm(0.25) → frame 500×640 (hop 320, 160 front pad) → symmetric
  Hann → rfft(1024) → magnitude / window.sum() → HTK mel (513→128, 60 Hz–16 kHz, DC bin
  zeroed) → `0.1·log(max(mel, 1e-5))`. Full spec in `perch2_logmel_settings.md`.
- **Stem uses VALID padding** (→ 249×63); all other k>1 convs use JAX `SAME`. Getting this
  wrong drops embedding cosine to ~0.82. Recovered from SE pooling divisors (15687 = 249×63).
- **Backbone:** EfficientNet-B3, 26 MBConv blocks, folded BatchNorm, Swish, sigmoid-gated SE,
  1536-d output, global average pool. 32 kHz, 5 s / 160000-sample windows.

## CRITICAL: the low-amplitude fix (read before touching frontend or adapter)

Perch peak-normalizes each window to 0.25 internally. On very low-amplitude input (MARS
hydrophone audio ~peak 0.002) that applies a large (~125×) amplification, and the port's
frontend arithmetic then diverges from live TF (cosine drops to 0.43–0.95 on quiet clips;
loud clips are fine at cos 1.0).

**Fix (already in `perch_hoplite_torch_adapter.py`):** `peak_normalize_windows()` peak-
normalizes every 5 s window to 0.25 *before* the model, inside `embed()`. It is **idempotent**
with Perch's internal peak-norm, so it yields the identical canonical embedding while keeping
the arithmetic numerically stable. Verified: real MARS windows go cos 0.76–0.94 → 1.00000.

Rules:
- Any audio→embedding path MUST peak-normalize each window to 0.25 first. Keep it in the one
  helper so embed and detect paths can't drift.
- Do NOT remove it thinking "peak_norm is internal already" — the internal one is what
  diverges at low amplitude. The pre-normalization is the fix.
- Do NOT call hoplite's `normalize_audio` (it subtracts the mean; Perch does not).

## Weights: regenerate, never commit

Weights come from the `justinchuby/Perch-onnx` export (ONNX stores conv weights in PyTorch
`OIHW` with BN pre-folded — what the code expects — and parses TF-free), validated end-to-end
against live TF at cosine ~1.0 with **relative L2 ~1e-5** (magnitude-faithful, not just
angle). A tensor-by-tensor weight match only hits ~52/441 — expected, not a bug: the rest are
ONNX's constant-folded / BN-fused combinations. Fidelity is established end-to-end.

## Repo layout

- Top level: core deliverable (`perch_frontend_torch.py`, `perch_embedder_torch.py`,
   `extract_weights.py`, `perch_hoplite_torch_adapter.py`, 
  `perch_bench_native_spark.py`, `check_adapter.py`, `check_db.py`), install/resample scripts,
  and docs (`README.md`, `perch2_logmel_settings.md`, `benchmark_results.md`).
- `dev/`: the reverse-engineering trail (introspect → onnx probe → frontend calibration →
  onnx2torch dead-end → benchmark). Not needed to run; it's the "how" story. See `dev/README.md`.

## Conventions

- Prose docs: minimal formatting. Distinguish proven (measured) vs intended vs unverified.
- Credit `justinchuby/Perch-onnx` and Google as prior art / upstream, not this project's work.
- Join embeddings to audio by recording **filename** (and window offsets) — glob order is
  arbitrary, not manifest order.
- Perch is amplitude-invariant per window by design — but very low absolute amplitude affects
  numerical stability (see the fix above).
- `new_32k_resample_sox.sh` has a `vol 3` gain: for *embeddings* it is a no-op (per-window
  peak-norm erases any constant gain). In the marine science pipeline it's retained as a
  voltage-calibration convention. Both statements are true; phrase per context.
