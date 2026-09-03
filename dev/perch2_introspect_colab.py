#!/usr/bin/env python3
# =============================================================================
# Perch 2.0 MODEL INTROSPECTION  --  RUN IN GOOGLE COLAB (same env as generator)
# =============================================================================
# Goal: reveal how perch_v2 is structured so we can plan the TF -> PyTorch port.
#   We need to know:
#     1. What kind of object model.embed() wraps (Keras Model? bare SavedModel?)
#     2. Whether we can isolate a spectrogram -> embedding sub-network
#     3. The full list of weight variables (names + shapes) for conversion
#     4. Total parameter count (should be ~12M for the EfficientNet-B3 embedder,
#        plus ~91M if the 14,795-class head is included)
#
# Nothing here changes anything; it only prints and writes a JSON report you can
# paste back. Run after the generator's install cells (perch-hoplite + TF 2.20).
# =============================================================================

import json, dataclasses
import numpy as np

from perch_hoplite.zoo import model_configs
model = model_configs.load_model_by_name('perch_v2')

report = {}

# ---- 1. The perch-hoplite wrapper object -----------------------------------
print("="*70)
print("WRAPPER OBJECT")
print("="*70)
print("type(model):", type(model))
wrapper_attrs = [a for a in dir(model) if not a.startswith("__")]
print("public attributes:", wrapper_attrs)
report["wrapper_type"] = str(type(model))
report["wrapper_attrs"] = wrapper_attrs

# ---- 2. Find the underlying TF/Keras object --------------------------------
# perch-hoplite wraps a Kaggle model; the real graph is usually held under one
# of these attribute names. We probe them without assuming which exists.
candidate_names = ["model", "_model", "tf_model", "_tf_model", "embedding_model",
                   "module", "_module", "loaded_model", "keras_model", "net"]
print("\n" + "="*70)
print("UNDERLYING MODEL CANDIDATES")
print("="*70)
underlying = None
for name in candidate_names:
    if hasattr(model, name):
        obj = getattr(model, name)
        print(f"  model.{name}: {type(obj)}")
        if underlying is None:
            underlying = obj
            report["underlying_attr"] = name
            report["underlying_type"] = str(type(obj))

if underlying is None:
    print("  (none of the common names found; inspect wrapper_attrs above)")

# ---- 3. SavedModel signatures (if it's a SavedModel) -----------------------
print("\n" + "="*70)
print("CALLABLE SIGNATURES (SavedModel)")
print("="*70)
sig_obj = underlying if underlying is not None else model
if hasattr(sig_obj, "signatures"):
    try:
        for sig_name, fn in sig_obj.signatures.items():
            print(f"  signature '{sig_name}':")
            si = getattr(fn, "structured_input_signature", None)
            so = getattr(fn, "structured_outputs", None)
            print("    inputs :", si)
            print("    outputs:", so)
        report["signatures"] = list(sig_obj.signatures.keys())
    except Exception as e:
        print("  error reading signatures:", e)
else:
    print("  no .signatures attribute (likely a Keras Model, not a bare SavedModel)")

# ---- 4. Keras layer structure (if it's a Keras Model) ----------------------
print("\n" + "="*70)
print("KERAS LAYERS (if applicable)")
print("="*70)
if hasattr(sig_obj, "layers"):
    try:
        for i, layer in enumerate(sig_obj.layers):
            shp = getattr(layer, "output_shape", "?")
            print(f"  [{i:3d}] {layer.__class__.__name__:24s} {getattr(layer,'name','')}  out={shp}")
        report["n_keras_layers"] = len(sig_obj.layers)
    except Exception as e:
        print("  error listing layers:", e)
else:
    print("  no .layers attribute")

# ---- 5. All weight variables (names + shapes) ------------------------------
# This is the map we need for weight conversion. Works for both Keras models
# (.weights / .variables) and SavedModels (.variables).
print("\n" + "="*70)
print("WEIGHT VARIABLES")
print("="*70)
vars_list = None
for attr in ["variables", "weights", "trainable_variables"]:
    obj = underlying if underlying is not None else model
    if hasattr(obj, attr):
        try:
            vs = getattr(obj, attr)
            if vs:
                vars_list = vs
                print(f"  using .{attr}  ({len(vs)} variables)")
                break
        except Exception:
            pass

var_report = []
total_params = 0
if vars_list is not None:
    for v in vars_list:
        name = getattr(v, "name", "?")
        shape = tuple(v.shape)
        n = int(np.prod(shape)) if all(s is not None for s in shape) else 0
        total_params += n
        var_report.append({"name": name, "shape": list(shape), "params": n})
    # print first 40 and last 10 so we see stem + head without flooding output
    for r in var_report[:40]:
        print(f"  {r['name']:60s} {str(r['shape']):22s} {r['params']:>10,}")
    if len(var_report) > 50:
        print(f"  ... ({len(var_report)-50} variables omitted) ...")
        for r in var_report[-10:]:
            print(f"  {r['name']:60s} {str(r['shape']):22s} {r['params']:>10,}")
    print(f"\n  TOTAL variables: {len(var_report)}")
    print(f"  TOTAL parameters: {total_params:,}")
else:
    print("  could not locate a variables/weights list on the model object")

report["n_variables"] = len(var_report)
report["total_params"] = int(total_params)
report["variables"] = var_report

# ---- 6. Save the full report -----------------------------------------------
with open("/content/perch2_structure_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nWrote /content/perch2_structure_report.json")
print("Paste back the printed output above (and/or download the JSON).")
