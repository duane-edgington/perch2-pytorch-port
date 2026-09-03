# Perch 2.0 GB10 benchmark

- onnxruntime 1.25.0, providers ['CUDAExecutionProvider', 'CPUExecutionProvider']
- base clip: clip00_input.npy, 5 s @ 32 kHz, peak-normed 0.25
- warmup 8, up to 40 timed iters / 8.0s per cell

### Latency (ms per run) — mean ± std (min)

| Model | Provider | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|---|
| perch_v2 (DFT) | CPU | 51.91 ± 5.15 (42.47) | 216.03 ± 7.81 (200.64) | 346.53 ± 10.09 (327.55) | 544.66 ± 14.69 (526.52) | 1067.82 ± 13.19 (1045.78) |
| perch_v2 (DFT) | CUDA(GB10) | 14.66 ± 1.10 (12.64) | 39.97 ± 1.25 (37.13) | 75.90 ± 1.35 (73.08) | 154.95 ± 1.45 (151.36) | 308.97 ± 1.85 (305.41) |
| perch_v2_no_dft | CPU | 45.30 ± 4.67 (37.88) | 193.59 ± 7.68 (173.44) | 305.01 ± 10.75 (290.86) | 477.62 ± 7.25 (466.35) | 952.67 ± 5.56 (938.89) |
| perch_v2_no_dft | CUDA(GB10) | 9.77 ± 0.91 (8.18) | 20.01 ± 1.13 (17.17) | 38.66 ± 1.32 (35.22) | 75.65 ± 1.12 (73.78) | 155.51 ± 0.95 (153.87) |

### Throughput (clips/sec)

| Model | Provider | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|---|
| perch_v2 (DFT) | CPU | 19.3 | 18.5 | 23.1 | 29.4 | 30.0 |
| perch_v2 (DFT) | CUDA(GB10) | 68.2 | 100.1 | 105.4 | 103.3 | 103.6 |
| perch_v2_no_dft | CPU | 22.1 | 20.7 | 26.2 | 33.5 | 33.6 |
| perch_v2_no_dft | CUDA(GB10) | 102.3 | 199.9 | 206.9 | 211.5 | 205.8 |

### Amortized latency per clip (ms/clip)

| Model | Provider | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|---|
| perch_v2 (DFT) | CPU | 51.91 | 54.01 | 43.32 | 34.04 | 33.37 |
| perch_v2 (DFT) | CUDA(GB10) | 14.66 | 9.99 | 9.49 | 9.68 | 9.66 |
| perch_v2_no_dft | CPU | 45.30 | 48.40 | 38.13 | 29.85 | 29.77 |
| perch_v2_no_dft | CUDA(GB10) | 9.77 | 5.00 | 4.83 | 4.73 | 4.86 |
