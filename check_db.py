import numpy as np, json
from perch_hoplite.db import sqlite_usearch_impl

mani = json.load(open("perch2_refs/manifest.json"))
src2clip = {c["source_file"]: c["clip_id"] for c in mani["clips"]}

db = sqlite_usearch_impl.SQLiteUSearchDB.create(db_path="./hoplite_db_test")
print("count:", db.count_embeddings(), "| dim:", db.get_embedding_dim(),
      "| dtype:", db.get_embedding_dtype())

for w in db.get_all_windows(include_embedding=True):
    v = np.asarray(w.embedding).reshape(-1).astype(np.float64)
    fn = db.get_recording(w.recording_id).filename           # e.g. 'dolphin_call_02.wav'
    clip = src2clip.get(fn)
    line = f"win {w.id}  {fn}  off {w.offsets}  norm={np.linalg.norm(v):.3f}"
    if clip:
        ref = np.load(f"perch2_refs/{clip}_embeddings.npy").reshape(-1).astype(np.float64)
        cos = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref)))
        line += f"  -> {clip}  cos={cos:.5f}  (ref norm {np.linalg.norm(ref):.3f})"
    print(line)
