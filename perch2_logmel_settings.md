# Perch 2.0 — Log-Mel Spectrogram Frontend Settings

These parameters were recovered by inspecting the Perch 2.0 ONNX export
(`justinchuby/Perch-onnx`) and cross-checking against the
`google-research/perch` source code. They are **not fully documented** by
Google — several were reverse-engineered.

Source: Duane R. Edgington, MBARI, July 2026.
GitHub: https://github.com/duane-edgington/perch-pytorch

---

## Audio Input

| Parameter | Value | Notes |
|---|---|---|
| Sample rate | 32,000 Hz | Must resample to 32 kHz before embedding |
| Window duration | 5.0 seconds | Fixed — model processes 5s clips only |
| Samples per window | 160,000 | 5.0 × 32,000 |
| Channels | Mono | Multi-channel audio should be downmixed |

---

## STFT Parameters

| Parameter | Value | Notes |
|---|---|---|
| Window length (KERNEL) | 640 samples | 20 ms at 32 kHz |
| Hop length (STRIDE) | 320 samples | 10 ms at 32 kHz |
| FFT size (NFFT) | 1,024 | → 513 frequency bins (rfft) |
| Time frames per window | 500 | After front-padding; `x.unfold(-1, 640, 320)[:, :500, :]` |
| Front pad | 160 samples | Half-hop pre-padding for exact frame origin alignment |
| Window function | Hann (symmetric) | `torch.hann_window(640, periodic=False)` |
| Magnitude normalization | divide by `window.sum()` | Power=1 (magnitude, not power spectrum) |

---

## Mel Filterbank Parameters

| Parameter | Value | Notes |
|---|---|---|
| Number of mel bands | 128 | |
| Frequency floor | 60 Hz | Low-frequency cutoff |
| Frequency ceiling | 16,000 Hz | Half the sample rate |
| Mel scale | HTK | `fmel = 2595·log10(1 + f/700)` |
| DC bin | Zeroed | Bin 0 set to 0 after filterbank |

---

## Log Scaling  ← **Critical: NOT PCEN**

```python
log_mel = 0.1 * log(max(mel_spectrogram, 1e-5))
```

| Parameter | Value | Notes |
|---|---|---|
| Scaling function | Log | `0.1 · log(max(x, floor))` |
| Log floor | 1e-5 | Prevents log(0) |
| Log scale factor | 0.1 | Compresses dynamic range |
| **NOT PCEN** | — | Perch 1.0 / SurfPerch use PCEN; Perch 2.0 uses log |

> **This is the most critical undocumented detail.** Perch 1.0 and SurfPerch
> use Per-Channel Energy Normalization (PCEN). Perch 2.0 uses simple log
> scaling. Using PCEN instead of log produces completely different embeddings.

---

## Numerical Precision

| Parameter | Value | Notes |
|---|---|---|
| Frontend computation | float64 | Required for near-exact parity |
| Model compute | float32 | The TF/PyTorch model computes in float32 |
| Embedding storage | float16 | Embeddings are stored as float16 **in the hoplite USearch index** — a storage/index choice, NOT the model's output dtype |

Running the frontend in float32 degrades cosine similarity on quiet recordings
(near the log floor) from ~1×10⁻⁴ to ~1×10⁻³ relative error. Running float64
throughout the frontend matches the reference to ~1×10⁻⁴.

---

## Feature Map After Stem Convolution

The EfficientNet-B3 stem uses **VALID padding** (no zero-padding before the
3×3 convolution). All subsequent convolutions use JAX-style SAME padding.

| Stage | Shape | Notes |
|---|---|---|
| STFT output | 500 × 513 | time frames × FFT bins (rfft of 1024) |
| Mel spectrogram output | 500 × 128 | time frames × mel bands |
| After stem (VALID pad) | 249 × 63 | VALID padding shrinks spatial dims; 15,687 = 249 × 63 (recoverable from SqueezeExcite pooling divisors in ONNX graph) |
| Final embedding | 1,536-dim | After EfficientNet-B3 + global average pool |

> Using SAME padding instead of VALID on the stem produces embedding cosine
> similarity of ~0.82 vs the reference — embeddings are numerically wrong.

---

## Amplitude Normalization — Universal Finding (applies to all sources)

**What Perch 2.0 does to the input (the answer to "what amplification/normalization?"):**
Perch peak-normalizes **each 5-second window** to a target peak of **0.25**:

```
x_normalized = x * (0.25 / max(|x|))
```

No fixed gain, no RMS normalization, and **no mean/DC subtraction** — purely dividing by the
window's peak absolute amplitude and rescaling so the loudest sample is 0.25. It is therefore
**amplitude-invariant**: any constant gain applied beforehand (e.g. SoX `vol 3`) is divided
straight back out and does not change the embedding. It is computed **per window**, on each
5 s clip independently.

For typical above-water bird recordings this internal step is effectively benign (raw peak
~0.5–1.0, so the rescale factor is ~0.25–0.5×). For the MARS hydrophone at 891 m depth,
typical peak amplitudes are 0.001–0.003 — three orders of magnitude quieter — so the internal
peak-norm applies a ~100–250× amplification. At that amplification the frontend arithmetic
becomes numerically sensitive and PyTorch diverges from TF unless the window is pre-normalized
(idempotent with the internal step) in higher precision first.

**Without pre-normalization:** PyTorch vs TF cosine similarity = 0.43–0.94 on
MARS audio.

**With pre-normalization:** cosine similarity ≈ 1.0 vs live TF on MARS windows
(GB10 GPU shows ~0.9999997, a float32/tensor-core artifact; CPU float64 reaches 1.0000000).

```python
def peak_normalize_window(audio, target_peak=0.25):
    peak = np.abs(audio).max()
    if peak > 1e-8:
        audio = audio * (target_peak / peak)
    return audio
```

### Amplitude-invariance is only APPROXIMATE — normalize ALWAYS, not just for quiet audio

Perch peak-normalizes internally, so in exact arithmetic it would be perfectly
amplitude-invariant (a constant gain washes out). Empirically it is **not** exact: the
invariance degrades smoothly as absolute input level drops, and the degradation begins well
above the extreme-quiet regime. Measured against a peak-0.25 reference (live TF, clip02, no
clipping — all inputs scaled *down* only):

| Input peak | cos vs peak-0.25 reference |
|---|---|
| 0.25  | 1.000000 |
| 0.125 | 0.998476 |
| 0.062 | 0.993660 |
| 0.025 | 0.981521 |

Even at peak 0.025 — only ~10× below the target, nowhere near hydrophone levels — embeddings
already differ by ~2% cosine. Below that (MARS at ~0.002) it degrades to cos 0.43–0.94. The
cause is numerical: the internal peak-norm amplifies quieter inputs more (×2, ×4, ×10…), and
the frontend's finite-precision FFT / mel matmul / near-`1e-5`-floor log do not perfectly
commute with that amplification.

**Conclusion (revised, stronger than "only for quiet audio"):** **pre-normalize every
5-second window to peak 0.25 before embedding, regardless of source.** It costs nothing —
it is idempotent with Perch's internal peak-norm, and at peak 0.25 the match to TF is exact
(cos 1.000000). It removes an amplitude-dependent inconsistency that is present even for
moderately quiet or level-varying recordings. This matters especially when a corpus mixes
recording levels (varying distance, gain, or source): without normalization, level
differences leak into the embedding and can separate sounds that are acoustically identical
— bad for clustering, search, and topic modeling; with it, the same sound at different levels
embeds consistently.

Validated against live TF Perch 2.0 (Colab, TF 2.20, GB10): documented frontend parameters
reproduce TF's `spectrogram` output at **cosine ≥ 0.9999998** across test clips (after
per-window normalization to 0.25).

---

## Summary — Implementation Checklist

For anyone reimplementing Perch 2.0's frontend from scratch:

- [ ] Resample audio to **32 kHz mono**
- [ ] Cut into **5-second windows** (160,000 samples), non-overlapping
- [ ] **Peak-normalize** each window to 0.25 — **always** (invariance is only approximate; see Amplitude section)
- [ ] STFT: **window=640, hop=320, FFT=1024**, symmetric Hann window, divide magnitude by `window.sum()`, front-pad 160 samples
- [ ] Mel filterbank: **128 bands, 60 Hz–16 kHz, HTK scale, zero DC bin**
- [ ] Log scaling: `0.1 · log(max(mel, 1e-5))` — **NOT PCEN**
- [ ] Run frontend in **float64**
- [ ] EfficientNet-B3 stem: **VALID padding** (3×3 conv, no zero-pad)
- [ ] All other convolutions: **SAME padding** (JAX convention)
- [ ] Output: **1536-dim embedding** per 5-second window (model computes float32; hoplite stores float16 in its index)

---

## References

- **Perch 2.0 model (the correct model citation):** van Merriënboer, B., Dumoulin, V.,
  Hamer, J., Harrell, L., Burns, A., Denton, T. (2025). "Perch 2.0: The Bittern Lesson
  for Bioacoustics." arXiv:2508.04665. https://arxiv.org/abs/2508.04665
- **Marine transfer-learning evaluation (a different paper — not the model citation):**
  Burns, A., Harrell, L., van Merriënboer, B., Dumoulin, V., Hamer, J., Denton, T. (2025).
  "Perch 2.0 transfers 'whale' to underwater tasks." NeurIPS 2025 Workshop: AI for
  Non-Human Animal Communication. arXiv:2512.03219. https://arxiv.org/abs/2512.03219
- **Agile modeling method:** Dumoulin et al. (2025), "The Search for Squawk: Agile Modeling
  in Bioacoustics."
- **perch-hoplite (system):** https://github.com/google-research/perch-hoplite
- **ONNX export used for weight extraction / validation:** https://huggingface.co/justinchuby/Perch-onnx
- **This PyTorch implementation:** https://github.com/duane-edgington/perch-pytorch
- **perch-hoplite pipeline (TF-free fork):** https://github.com/duane-edgington/perch-hoplite

> **Citation caution for colleagues:** the `google-research/perch` GitHub repo's main README
> states "Model frontend - we use a PCEN melspectrogram" — but that describes **Perch 1.0**,
> the currently-released Perch in that repo's overview, **not Perch 2.0**. This is a primary
> source of the PCEN confusion. Perch 2.0 (arXiv:2508.04665) uses log-mel, confirmed by the
> paper, by this reverse-engineering, and directly from the live TF model's spectrogram
> output (see below).

---

*Duane R. Edgington — MBARI — July 2026*
