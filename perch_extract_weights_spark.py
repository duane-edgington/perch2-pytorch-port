#!/usr/bin/env python3
# =============================================================================
# Perch 2.0 EMBEDDER weight extraction  (RUN ON THE SPARK)
# =============================================================================
# Dumps the EfficientNet-B3 backbone weights (stem, 26 MBConv blocks, head conv,
# and their folded BatchNorm affines + SE weights) from the validated ONNX into
# a single npz, plus a full graph manifest (node op sequence + every initializer
# name/shape). The classification heads (huge 14795-class matmuls) are skipped
# to keep the upload small -- we only need up to the 1536-d embedding.
#
# Output: ~/perch-pytorch/perch_weights/{weights.npz, graph_manifest.json}
# Zip perch_weights/ and upload both files.
# =============================================================================
import os, sys, json
import numpy as np

os.system(f"{sys.executable} -m pip install -q onnx huggingface_hub 2>/dev/null")
import onnx
from onnx import numpy_helper
from huggingface_hub import hf_hub_download

OUT = os.path.expanduser("~/perch-pytorch/perch_weights")
os.makedirs(OUT, exist_ok=True)
SIZE_CAP = 3_000_000          # skip initializers bigger than this (= class heads)

path = hf_hub_download(repo_id="justinchuby/Perch-onnx", filename="perch_v2.onnx")
m = onnx.load(path)
g = m.graph
print(f"graph: {len(g.node)} nodes, {len(g.initializer)} initializers")

# ---- full node manifest (topology) ------------------------------------------
nodes = [dict(i=i, op=n.op_type, name=n.name,
              inputs=list(n.input), outputs=list(n.output))
         for i, n in enumerate(g.node)]

# ---- initializers: save backbone-sized ones to npz --------------------------
weights = {}
init_manifest = {}
skipped_big = []
for init in g.initializer:
    arr = numpy_helper.to_array(init)
    saved = arr.size <= SIZE_CAP
    init_manifest[init.name] = dict(shape=list(arr.shape), dtype=str(arr.dtype),
                                    size=int(arr.size), saved=bool(saved))
    if saved:
        weights[init.name] = arr
    else:
        skipped_big.append((init.name, list(arr.shape), int(arr.size)))

# Some Conv weights are stored as Constant NODES (not initializers). Capture those too.
for n in g.node:
    if n.op_type == "Constant":
        for attr in n.attribute:
            if attr.name == "value":
                arr = numpy_helper.to_array(attr.t)
                if 0 < arr.size <= SIZE_CAP and n.output:
                    nm = n.output[0]
                    if nm not in weights:
                        weights[nm] = arr
                        init_manifest[nm] = dict(shape=list(arr.shape),
                                                 dtype=str(arr.dtype),
                                                 size=int(arr.size), saved=True,
                                                 from_constant_node=True)

np.savez(os.path.join(OUT, "weights.npz"), **weights)
json.dump(dict(nodes=nodes, initializers=init_manifest,
               skipped_big=skipped_big, size_cap=SIZE_CAP),
          open(os.path.join(OUT, "graph_manifest.json"), "w"))

tot = sum(a.size for a in weights.values())
sz = os.path.getsize(os.path.join(OUT, "weights.npz")) / 1e6
print(f"saved {len(weights)} weight tensors ({tot/1e6:.1f}M params) -> weights.npz ({sz:.1f} MB)")
print(f"skipped {len(skipped_big)} big tensors (class heads): "
      f"{[s[0] for s in skipped_big][:6]}")
print(f"\nUpload BOTH: {OUT}/weights.npz  and  {OUT}/graph_manifest.json")
print("(zip the perch_weights folder if easier)")
