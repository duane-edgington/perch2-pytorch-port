#!/usr/bin/env python3
# =============================================================================
# perch-hoplite adapter for the native PyTorch Perch 2.0 model
# =============================================================================
# Lets you run the WHOLE perch-hoplite workflow on the Spark in pure PyTorch:
# embedding generation (this file) + search/active-learning/classifier (hoplite's
# own agile tools) -- no TensorFlow, no Colab round-trip.
#
# perch-hoplite's core (DB + search + agile pipeline) does NOT require TensorFlow;
# TF/JAX/ONNX are optional extras. Install core only:
#     pip install perch-hoplite            # (no [tf]/[jax] extras)
#
# This wraps PerchModel (perch_embedder_torch.py + perch_frontend_torch.py) in
# hoplite's zoo_interface.EmbeddingModel, then uses hoplite's EmbedWorker to
# populate a SQLite+USearch DB -- the same DB your existing agile search/
# classifier notebooks read. Swap this in where you previously used the TF model.
#
# Usage:
#     python perch_hoplite_torch_adapter.py \
#         --audio_dir /path/to/wavs --glob '*.wav' \
#         --db_dir ./hoplite_db --weights_dir ./perch_weights \
#         --exact_mel ./const__pad1_output_0.npy --device cuda
# Then point the agile search/classifier UI at ./hoplite_db as usual.
# =============================================================================
import dataclasses
import numpy as np
import torch

from perch_hoplite.zoo import zoo_interface
from perch_embedder_torch import PerchModel

SR = 32000
WINDOW_S = 5.0          # Perch embeds 5 s windows (160000 samples)
EMB_DIM = 1536
PEAK_TARGET = 0.25      # Perch's internal peak-norm target; pre-normalizing to it keeps
                        # the log-mel frontend numerically stable at low input amplitude.


def peak_normalize_windows(frames: np.ndarray, target_peak: float = PEAK_TARGET) -> np.ndarray:
    """Per-window peak normalization to `target_peak`, computed in float64.

    Idempotent with Perch's internal peak_norm(0.25): yields the identical canonical,
    amplitude-invariant embedding, but keeps the log-mel frontend numerically stable so
    the port matches the reference TF model (cos 1.0) even on very low-amplitude input
    (e.g. MARS hydrophone audio near peak 0.002). Silent windows are left unscaled.
    """
    f = np.atleast_2d(frames).astype(np.float64)
    peak = np.abs(f).max(axis=-1, keepdims=True)
    scale = np.where(peak > 1e-12, target_peak / np.maximum(peak, 1e-12), 1.0)
    return (f * scale).astype(np.float32)

@dataclasses.dataclass
class PerchTorchModel(zoo_interface.EmbeddingModel):
    """Native-PyTorch Perch 2.0 embedder, as a hoplite EmbeddingModel.

    hoplite passes a (possibly long) audio array to embed(); we frame it into
    5 s windows, run them through the native model, and return embeddings shaped
    [Frames, Channels=1, Features=1536] as the interface requires.
    """
    sample_rate: int = SR          # SR = 32000, defined at top of file
    weights_dir: str = "perch_weights"
    exact_mel: str | None = None
    device: str = "cuda"
    window_size_s: float = WINDOW_S
    hop_size_s: float = WINDOW_S     # non-overlapping; set < WINDOW_S to overlap
    embedding_dim: int = EMB_DIM
    use_compile: bool = False

    def __post_init__(self):
        dev = self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.device = dev
        self._model = PerchModel(self.weights_dir, exact_mel_npy=self.exact_mel).eval().to(dev)
        # Blackwell tensor cores for float32 (harmless, faster). Comment out for max fidelity.
        if dev == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # torch.compile gives ~2.5x throughput on GB10 at batch>=4.
        # First batch is slow (graph compilation); pays off for >~50 files.
        if self.use_compile:
            self._model = torch.compile(self._model)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @torch.no_grad()
    def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
        # Frame to [Frames, 160000]. NOTE: we do NOT call hoplite's normalize_audio
        # (which subtracts the mean). Instead we pre-peak-normalize each window to 0.25
        # (idempotent with Perch's internal peak_norm) so the log-mel frontend stays
        # numerically stable and matches the TF reference at cos 1.0 even on low-amplitude
        # input. Perch's internal peak_norm alone matches TF only at healthy amplitude;
        # on quiet input (e.g. MARS audio near peak 0.002) it diverges without this step.
        framed = self.frame_audio(audio_array, self.window_size_s, self.hop_size_s)
        if framed.ndim == 1:
            framed = framed[None, :]
        framed = peak_normalize_windows(framed)          # low-amplitude fix: per-window peak-norm to 0.25
        x = torch.from_numpy(np.ascontiguousarray(framed)).float().to(self.device)
        emb = self._model(x).float().cpu().numpy()      # [Frames, 1536]
        emb = emb[:, np.newaxis, :]                     # [Frames, Channels=1, Features]
        return zoo_interface.InferenceOutputs(embeddings=emb)

    @torch.no_grad()
    def batch_embed(self, audio_batch: np.ndarray) -> zoo_interface.InferenceOutputs:
        outs = [self.embed(a).embeddings for a in audio_batch]
        return zoo_interface.InferenceOutputs(embeddings=np.stack(outs, axis=0), batched=True)


def build_db(audio_dir, file_glob, db_dir, weights_dir, exact_mel, device,
             dataset_name="spark", batch_size=32, hop_size_s=WINDOW_S,
             handle_duplicates="skip", use_compile=False):
    """Embed an audio folder into a hoplite DB using the native PyTorch model.

    hop_size_s: window hop in seconds. WINDOW_S (5.0) = non-overlapping; set
      smaller (e.g. 2.5) for overlapping windows / finer temporal search
      resolution. Choose this BEFORE a big run -- re-embedding is the costly step.
    handle_duplicates: 'skip' makes re-runs idempotent (append new audio without
      erroring on already-embedded files); 'error' is hoplite's strict default.
    """
    from ml_collections import config_dict
    from perch_hoplite.db import sqlite_usearch_impl
    from perch_hoplite.agile import embed as agile_embed
    from perch_hoplite.agile import source_info

    model = PerchTorchModel(weights_dir=weights_dir, exact_mel=exact_mel,
                            device=device, hop_size_s=hop_size_s,
                            use_compile=use_compile)

    # DB (SQLite + USearch index), IP metric, dim 1536
    usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(EMB_DIM)
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(db_path=db_dir, usearch_cfg=usearch_cfg)

    # Audio source glob
    glob_cfg = source_info.AudioSourceConfig(
        dataset_name=dataset_name, base_path=audio_dir, file_glob=file_glob,
        min_audio_len_s=1.0, target_sample_rate_hz=SR, shard_len_s=60.0)
    audio_sources = source_info.AudioSources(audio_globs=(glob_cfg,))

    # model_config: model_key is unused because we pass embedding_model directly,
    # but embedding_dim MUST match (hoplite validates it against the DB).
    model_config = agile_embed.ModelConfig(
        model_key="perch_torch", embedding_dim=EMB_DIM, model_config=config_dict.ConfigDict())

    worker = agile_embed.EmbedWorker(
        audio_sources=audio_sources, model_config=model_config, db=db,
        embedding_model=model)          # <-- the hook: our PyTorch model
    worker.process_all(batch_size=batch_size, handle_duplicates=handle_duplicates)
    db.commit()
    print(f"Done. {db.count_embeddings()} embeddings in {db_dir}")
    return db


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--glob", default="*.wav")
    ap.add_argument("--db_dir", default="./hoplite_db")
    ap.add_argument("--weights_dir", default="./perch_weights")
    ap.add_argument("--exact_mel", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset_name", default="spark")
    ap.add_argument("--hop_size_s", type=float, default=WINDOW_S,
                    help="window hop in seconds; 5.0=non-overlapping, e.g. 2.5 to overlap")
    ap.add_argument('--compile', action='store_true', default=False,
                    help='Enable torch.compile (~2.5x faster on GB10, slow first batch)')
    a = ap.parse_args()
    build_db(a.audio_dir, a.glob, a.db_dir, a.weights_dir, a.exact_mel,
             a.device, a.dataset_name, hop_size_s=a.hop_size_s,
             use_compile=a.compile)






