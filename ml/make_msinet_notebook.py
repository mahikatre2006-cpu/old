"""Generates ml/notebooks/msinet_to_onnx.ipynb (run once; the notebook itself is what you open in Colab)."""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


cells = [
    md("""
# MSI-Net → ONNX (for the thumbnail heatmap backend)

Converts the pretrained MSI-Net SALICON model (`alexanderkroner/MSI-Net` on Hugging Face, MIT licence) to a
fixed-size ONNX file, then checks the ONNX output matches TensorFlow.

**Run all cells top to bottom (CPU runtime is fine). The last cell downloads `msinet.onnx`.**
Put that file in `C:\\epoch1\\backend\\models\\`.
"""),
    code("""
!pip -q install tf2onnx onnx onnxruntime huggingface_hub
import tensorflow as tf, numpy as np, onnxruntime as ort, tf2onnx, sys
print("python", sys.version.split()[0], "| tf", tf.__version__, "| numpy", np.__version__, "| tf2onnx", tf2onnx.__version__)
"""),
    md("## 1. Download the model"),
    code("""
from huggingface_hub import snapshot_download
hf_dir = snapshot_download(repo_id="alexanderkroner/MSI-Net")
!ls -la {hf_dir} {hf_dir}/variables
"""),
    md("""
## 2. Load it and look at its signature

`tf.saved_model.load` is used instead of `keras.models.load_model` because newer Colab images ship Keras 3,
which cannot load this legacy SavedModel directory.
"""),
    code("""
m = tf.saved_model.load(hf_dir)
print("signatures:", list(m.signatures.keys()))
sig = m.signatures["serving_default"]
print("inputs :", sig.structured_input_signature)
print("outputs:", sig.structured_outputs)
IN_NAME = list(sig.structured_input_signature[1].keys())[0]
# the saved signature's output name is not necessarily "output", so look it up (prefer the 4-D map)
outs = sig.structured_outputs
OUT_NAME = next((k for k, v in outs.items() if len(v.shape) == 4), list(outs.keys())[0])
print("input arg name:", IN_NAME, "| output name:", OUT_NAME)
"""),
    md("""
## 3. Convert to ONNX

Fixed input `[1, 240, 320, 3]` float32, raw RGB values 0–255 (MSI-Net's own landscape size; the backend
letterboxes 16:9 thumbnails into it exactly like the model's README does). Output `[1, 240, 320, 1]`.
"""),
    code("""
import tf2onnx

@tf.function(input_signature=[tf.TensorSpec([1, 240, 320, 3], tf.float32, name="input")])
def fn(x):
    return {"output": sig(**{IN_NAME: x})[OUT_NAME]}

try:
    tf2onnx.convert.from_function(fn, input_signature=fn.input_signature, opset=13, output_path="msinet.onnx")
    print("converted with from_function")
except Exception as e:
    print("from_function failed:", repr(e)[:300], "\\n-> falling back to the CLI converter")
    !python -m tf2onnx.convert --saved-model {hf_dir} --signature_def serving_default --tag serve --opset 13 --output msinet.onnx
!ls -la msinet.onnx
"""),
    md("""
## 4. Check ONNX against TensorFlow on the model's example image

Pass criteria: correlation ≈ 1.0 and max abs difference around 1e-4 or smaller.
"""),
    code("""
import matplotlib.pyplot as plt

def preprocess(img, target=(240, 320)):
    t = tf.image.resize(tf.expand_dims(img, 0), target, preserve_aspect_ratio=True)
    vp, hp = target[0] - t.shape[1], target[1] - t.shape[2]
    pad = [[0, 0], [vp // 2, vp - vp // 2], [hp // 2, hp - hp // 2], [0, 0]]
    return tf.pad(t, pad).numpy().astype(np.float32), pad

img = np.array(tf.keras.utils.load_img(hf_dir + "/example.jpg"), dtype=np.float32)
x, pad = preprocess(img)
print("model input", x.shape, "range", x.min(), x.max())

ref = sig(**{IN_NAME: tf.constant(x)})[OUT_NAME].numpy()

sess = ort.InferenceSession("msinet.onnx", providers=["CPUExecutionProvider"])
print("onnx inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("onnx outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
out = sess.run(None, {sess.get_inputs()[0].name: x})[0]

cc = np.corrcoef(ref.ravel(), out.ravel())[0, 1]
print(f"output range {out.min():.5f}..{out.max():.5f} | sum {out.sum():.3f}")
print(f"corr(TF, ONNX) = {cc:.6f} | max abs diff = {np.abs(ref - out).max():.2e}")
assert cc > 0.999, "ONNX output does not match TensorFlow - do not use this file"

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].imshow(img / 255); ax[0].set_title("input"); ax[1].imshow(ref[0, ..., 0], cmap="inferno"); ax[1].set_title("TensorFlow")
ax[2].imshow(out[0, ..., 0], cmap="inferno"); ax[2].set_title("ONNX")
[a.axis("off") for a in ax]; plt.show()
"""),
    md("## 5. Download the file"),
    code("""
import hashlib
data = open("msinet.onnx", "rb").read()
print("size", f"{len(data)/1e6:.1f} MB", "| sha256", hashlib.sha256(data).hexdigest())
from google.colab import files
files.download("msinet.onnx")
"""),
]

nb = {
    "cells": cells,
    "metadata": {"colab": {"name": "msinet_to_onnx.ipynb"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).parent / "notebooks" / "msinet_to_onnx.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
