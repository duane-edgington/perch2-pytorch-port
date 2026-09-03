import numpy as np, torch
from perch_hoplite_torch_adapter import PerchTorchModel

m = PerchTorchModel(weights_dir="./perch_weights",
                    exact_mel="./const__pad1_output_0.npy", device="cuda")

for cid in ["clip00","clip01","clip02","clip03"]:
    x = np.load(f"perch2_refs/{cid}_input.npy")
    out = m.embed(x)
    emb = out.embeddings.reshape(-1)
    ref = np.load(f"perch2_refs/{cid}_embeddings.npy").reshape(-1)
    cos = float(emb @ ref / (np.linalg.norm(emb) * np.linalg.norm(ref)))
    print(f"{cid}: frames={out.embeddings.shape[0]} shape={out.embeddings.shape} cos={cos:.7f}")
