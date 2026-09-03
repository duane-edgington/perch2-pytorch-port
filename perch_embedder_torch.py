#!/usr/bin/env python3
# =============================================================================
# Perch 2.0 embedder — native PyTorch reimplementation (VALIDATED, cos=1.0)
# =============================================================================
# A from-scratch EfficientNet-B3 embedder matching Google Perch 2.0, with weights
# loaded from the model's own graph (extracted via extract_weights.py
# -> weights.npz + graph_manifest.json). Combined with PerchFrontend it forms a
# fully TensorFlow-free, ONNX-free native PyTorch Perch 2.0 embedding model.
#
# Validated against Google's reference TF outputs:
#   embedder alone (reference frontend -> embedding): cos 0.9999999, rel err ~7e-7
#   full pipeline  (raw audio -> frontend -> embedder): cos 1.0000000, rel err ~1-5e-5
#
# Architecture (read from the graph, matches chirp EfficientNet B3 exactly):
#   stem 3x3 s2 VALID -> 40ch -> 26 MBConv blocks -> head 1x1 -> 1536ch
#   MBConv = [expand 1x1] -> depthwise kxk -> SqueezeExcite -> project 1x1, folded
#   BatchNorm (scale/bias), Swish activation; SE gate uses sigmoid.
#   stem uses VALID padding (-> 249x63); all other k>1 convs use JAX 'SAME'.
#   embedding = global average pool of the (16,4,1536) spatial map.
#
# Usage:
#   from perch_embedder_torch import PerchModel
#   model = PerchModel("path/to/weights_dir").eval()   # dir has weights.npz + graph_manifest.json
#   emb = model(audio_1x160000)                         # (B,1536)
# =============================================================================
import os, json, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE2_BLOCKS = {2, 5, 8, 18}   # first block of stages 2,3,4,6


def _same_pad(x, k, s):
    ih, iw = x.shape[-2:]
    ph = max((-(-ih // s) - 1) * s + k - ih, 0)
    pw = max((-(-iw // s) - 1) * s + k - iw, 0)
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))  # JAX 'SAME': extra at end


class _Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class _Conv(nn.Module):
    def __init__(self, w, b, stride, groups=1, valid=False):
        super().__init__()
        self.w = nn.Parameter(w, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False) if b is not None else None
        self.k, self.s, self.g, self.valid = w.shape[-1], stride, groups, valid

    def forward(self, x):
        if not self.valid:
            x = _same_pad(x, self.k, self.s)
        return F.conv2d(x, self.w, self.b, self.s, 0, 1, self.g)


class _BN(nn.Module):   # folded BatchNorm: per-channel affine
    def __init__(self, scale, bias):
        super().__init__()
        self.s = nn.Parameter(scale.view(1, -1, 1, 1), requires_grad=False)
        self.b = nn.Parameter(bias.view(1, -1, 1, 1), requires_grad=False)

    def forward(self, x):
        return x * self.s + self.b


class _SE(nn.Module):
    def __init__(self, rw, rb, ew, eb):
        super().__init__()
        self.rw = nn.Parameter(rw, requires_grad=False); self.rb = nn.Parameter(rb.view(-1), requires_grad=False)
        self.ew = nn.Parameter(ew, requires_grad=False); self.eb = nn.Parameter(eb.view(-1), requires_grad=False)
        self.act = _Swish()

    def forward(self, x):
        s = x.mean((2, 3))
        s = self.act(s @ self.rw + self.rb)          # reduce  (C -> r)
        s = torch.sigmoid(s @ self.ew + self.eb)     # expand  (r -> C), sigmoid gate
        return x * s.unsqueeze(-1).unsqueeze(-1)


class _MBConv(nn.Module):
    def __init__(self, store, cfg):
        super().__init__()
        i = cfg["idx"]; p = f"MBConv_{i}|"
        self.res = (cfg["stride"] == 1 and cfg["in_ch"] == cfg["out_ch"])
        self.act = _Swish()
        self.expand = None
        if cfg["has_expand"]:
            self.expand = _Conv(store[p + "ExpandConv|w"], store.get(p + "ExpandConv|b"), 1)
            self.ebn = _BN(store[p + "ExpandBatchNorm|scale"], store[p + "ExpandBatchNorm|bias"])
        self.dw = _Conv(store[p + "DepthwiseConv|w"], store.get(p + "DepthwiseConv|b"),
                        cfg["stride"], groups=cfg["exp_out"])
        self.dbn = _BN(store[p + "DepthwiseBatchNorm|scale"], store[p + "DepthwiseBatchNorm|bias"])
        self.se = _SE(store[p + "SqueezeAndExcitation_0/Reduce|w"], store[p + "SqueezeAndExcitation_0/Reduce|b"],
                      store[p + "SqueezeAndExcitation_0/Expand|w"], store[p + "SqueezeAndExcitation_0/Expand|b"])
        self.proj = _Conv(store[p + "ProjectConv|w"], store.get(p + "ProjectConv|b"), 1)
        self.pbn = _BN(store[p + "ProjectBatchNorm|scale"], store[p + "ProjectBatchNorm|bias"])

    def forward(self, x):
        inp = x
        if self.expand is not None:
            x = self.act(self.ebn(self.expand(x)))
        x = self.act(self.dbn(self.dw(x)))
        x = self.se(x)
        x = self.pbn(self.proj(x))
        return x + inp if self.res else x


def _parse_graph(weights_dir):
    """Extract semantic weight tensors from weights.npz + graph_manifest.json."""
    M = json.load(open(os.path.join(weights_dir, "graph_manifest.json")))
    W = np.load(os.path.join(weights_dir, "weights.npz"))
    wset = set(W.keys()); nodes = M["nodes"]
    out2node = {n["outputs"][0]: n for n in nodes if n["outputs"]}
    t = lambda name: torch.from_numpy(W[name].astype(np.float32))
    pathof = lambda n: (n["outputs"][0] if n["outputs"] else "")
    store = {}

    def modtag(p):
        m = re.search(r"(Stem_0|MBConv_\d+)/(ExpandConv|DepthwiseConv|ProjectConv|"
                      r"SqueezeAndExcitation_0/Reduce|SqueezeAndExcitation_0/Expand)", p)
        return f"{m.group(1)}|{m.group(2)}" if m else None

    for n in nodes:
        if n["op"] == "Conv":
            tag = modtag(n["inputs"][0]) or modtag(pathof(n))
            if tag is None and "Conv_85" in pathof(n):
                tag = "Head|Conv"
            if tag is None:
                continue
            store[f"{tag}|w"] = t(n["inputs"][1])
            if len(n["inputs"]) > 2 and n["inputs"][2] in wset:
                store[f"{tag}|b"] = t(n["inputs"][2])
        elif n["op"] == "MatMul":
            tag = modtag(pathof(n))
            if tag and "SqueezeAndExcitation" in tag:
                store[f"{tag}|w"] = t(n["inputs"][1])
        if n["op"] == "Add":
            m = re.search(r"(MBConv_\d+)/SqueezeAndExcitation_0/(Reduce|Expand)", pathof(n))
            if m:
                for i in n["inputs"]:
                    if i in wset and W[i].ndim <= 2 and W[i].size < 20000:
                        store[f"{m.group(1)}|SqueezeAndExcitation_0/{m.group(2)}|b"] = t(i)

    # stem conv (input is the unsqueezed spectrogram)
    for n in nodes:
        if n["op"] == "Conv" and any("broadcast_in_dim" in i for i in n["inputs"][:1]):
            store["Stem_0|Conv_0|w"] = t(n["inputs"][1])
            if len(n["inputs"]) > 2 and n["inputs"][2] in wset:
                store["Stem_0|Conv_0|b"] = t(n["inputs"][2])

    # folded BN affines by module path
    def bnkey(p):
        m = re.search(r"(Stem_0|MBConv_\d+)/(\w*BatchNorm\w*)/", p)
        return f"{m.group(1)}|{m.group(2)}" if m else None
    for n in nodes:
        if n["op"] in ("Mul", "Add"):
            k = bnkey(pathof(n))
            if not k:
                continue
            for i in n["inputs"]:
                if i in wset and 1 <= W[i].ndim and W[i].size <= 3000:
                    store[f"{k}|{'scale' if n['op']=='Mul' else 'bias'}"] = t(i).reshape(-1)

    # head BN: trace from spatial_embedding = Mul(bn_add, sigmoid)
    se = out2node.get("spatial_embedding")
    if se:
        for i in se["inputs"]:
            if i in out2node and out2node[i]["op"] == "Add":
                addn = out2node[i]
                for j in addn["inputs"]:
                    if j in wset:
                        store["Head|BatchNorm|bias"] = t(j).reshape(-1)
                for j in addn["inputs"]:
                    if j in out2node and out2node[j]["op"] == "Mul":
                        for kk in out2node[j]["inputs"]:
                            if kk in wset:
                                store["Head|BatchNorm|scale"] = t(kk).reshape(-1)

    # block configs from weight shapes
    cfgs = []
    for i in range(26):
        exp = store.get(f"MBConv_{i}|ExpandConv|w")
        dw = store[f"MBConv_{i}|DepthwiseConv|w"]
        proj = store[f"MBConv_{i}|ProjectConv|w"]
        cfgs.append(dict(idx=i, has_expand=exp is not None,
                         exp_out=(exp.shape[0] if exp is not None else dw.shape[0]),
                         in_ch=(exp.shape[1] if exp is not None else dw.shape[0]),
                         out_ch=proj.shape[0], stride=2 if i in STRIDE2_BLOCKS else 1))
    return store, cfgs


class PerchEmbedder(nn.Module):
    """EfficientNet-B3 embedder: spectrogram (B,500,128) -> embedding (B,1536)."""
    def __init__(self, weights_dir):
        super().__init__()
        store, cfgs = _parse_graph(weights_dir)
        self.stem = _Conv(store["Stem_0|Conv_0|w"], store.get("Stem_0|Conv_0|b"), 2, valid=True)
        self.sbn = _BN(store["Stem_0|BatchNorm_0|scale"], store["Stem_0|BatchNorm_0|bias"])
        self.act = _Swish()
        self.blocks = nn.ModuleList([_MBConv(store, c) for c in cfgs])
        self.head = _Conv(store["Head|Conv|w"], store.get("Head|Conv|b"), 1)
        self.hbn = _BN(store["Head|BatchNorm|scale"], store["Head|BatchNorm|bias"])

    def spatial(self, spec):
        x = spec.unsqueeze(1)                       # (B,1,500,128)
        x = self.act(self.sbn(self.stem(x)))
        for b in self.blocks:
            x = b(x)
        return self.act(self.hbn(self.head(x)))     # (B,1536,16,4)

    def forward(self, spec):
        return self.spatial(spec).mean((2, 3))      # (B,1536)


class PerchModel(nn.Module):
    """Full native Perch 2.0 embedding model: audio (B,160000) -> embedding (B,1536)."""
    def __init__(self, weights_dir, exact_mel_npy=None):
        super().__init__()
        from perch_frontend_torch import PerchFrontend
        self.frontend = PerchFrontend()
        if exact_mel_npy:
            self.frontend.load_exact_mel(exact_mel_npy)
        else:
            # Prefer the exact matrix from the weights archive; fall back to the
            # HTK reconstruction (agrees to <2e-5) if the key isn't present.
            if self.frontend.load_exact_mel_from_npz(weights_dir) is None:
                print("note: exact mel not found in weights.npz — "
                      "using HTK reconstruction (agrees to <2e-5)")
        self.embedder = PerchEmbedder(weights_dir)

    def forward(self, audio, return_spatial=False):
        spec = self.frontend(audio)                 # (B,500,128)
        if return_spatial:
            sp = self.embedder.spatial(spec)
            return sp.mean((2, 3)), sp, spec
        return self.embedder(spec)


if __name__ == "__main__":
    import sys
    wdir = sys.argv[1] if len(sys.argv) > 1 else "pw"
    refs = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/uploads"
    mel = os.path.join(refs, "const__pad1_output_0.npy")
    model = PerchModel(wdir, exact_mel_npy=mel if os.path.exists(mel) else None).eval()
    cos = lambda a, b: float(a.reshape(-1) @ b.reshape(-1) /
                             (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    import glob
    print("Full native pipeline (raw audio -> embedding) vs TF reference:")
    with torch.no_grad():
        for ip in sorted(glob.glob(os.path.join(refs, "clip*_input.npy"))):
            cid = os.path.basename(ip).replace("_input.npy", "")
            ep = os.path.join(refs, f"{cid}_embeddings.npy")
            if not os.path.exists(ep):
                continue
            x = torch.from_numpy(np.load(ip))
            r = np.load(ep).reshape(-1)
            e = model(x).numpy().reshape(-1)
            print(f"  {cid}: cos={cos(e, r):.7f}  rel_err={np.linalg.norm(e-r)/np.linalg.norm(r):.3e}")
