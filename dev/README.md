# dev/ — how the port was reverse-engineered

These scripts are **not needed to run the port**. They are kept as the record of *how* an
undocumented TensorFlow model was turned into an idiomatic PyTorch reimplementation — the
"why and how" that may be the most useful part of this repo if you have never done it
yourself. They are exploratory: some are Colab scripts (with `!pip` magics, not runnable as
plain `.py`), some are Spark probes.

The story, in the order it happened:

1. **`perch2_introspect_colab.py`** — dump the TF model's structure: layer names, weight
   shapes, parameter counts, the serving signature. The starting point: *what is actually
   in this model?* (Colab; loads the TF model.)

2. **`perch2_to_onnx_probe_colab.py`** — convert the SavedModel to ONNX and run the first
   parity probe, comparing raw vs peak-normalized input. Early evidence about the frontend
   and where the numbers came from. (Colab; `!pip` magics — not plain-Python runnable.)

3. **`perch_frontend_probe_spark.py`** — strong-signal probes to calibrate the frontend
   (window, hop, FFT size, mel filterbank). **Historical note:** this early script assumed
   the frontend used PCEN — later disproved. Perch 2.0 uses `0.1·log(max(mel,1e-5))`, not
   PCEN (confirmed against the live TF spectrogram; the floor sits exactly at
   `0.1·ln(1e-5) = −1.15129`). Kept as-is to show the wrong turn and the correction.

4. **`perch_spark_onnx2torch.py`** — the attempt to convert the ONNX graph to PyTorch
   automatically via `onnx2torch`, plus an onnxruntime fallback. It documents the ops that
   would not convert in this environment — which is *why* the embedder was ultimately
   hand-built as an idiomatic `nn.Module` rather than a converted graph. (Other projects
   have since gotten onnx2torch to work; in this environment it did not, and the hand build
   is the readable result.)

5. **`perch_spark_benchmark.py`** — benchmark the ONNX model (both ONNX variants × both
   execution providers × batch sweep). Produced the ONNX-bridge throughput numbers the
   top-level README compares the native PyTorch path against.

The key details these probes recovered — and that the production code in the repo root
depends on — are written up in `../perch2_logmel_settings.md` (the verified frontend spec)
and summarized in the top-level `README.md` ("The architecture, as recovered from the
model"): log-mel not PCEN, the VALID-padding stem, the exact STFT/mel parameters, and the
per-window peak-normalization needed for exactness on quiet audio.
