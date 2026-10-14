"""Export rfdetr-vr-v1 core (768px, batch 2 — the VR pipeline configuration) to
ONNX opset 19, for the vs-mlrt Windows MIGraphX benchmark. Export only; no
timing here because the 8K baseline job currently owns the GPU."""
import sys
from pathlib import Path

import torch

WS = Path(r"E:\AI\ai_workspace\ShiMu\jasna_src")
sys.path.insert(0, str(WS))

import rfdetr  # noqa: E402

WEIGHTS = WS / "model_weights" / "rfdetr-vr-v1.pt"
RES = 768
BATCH = 2
ONNX_PATH = Path(r"E:\AI\ai_workspace\ShiMu\results\probe\rfdetr-vr-v1-b2.onnx")

device = torch.device("cuda:0")
checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
state = checkpoint["model"]
num_classes = int(state["class_embed.weight"].shape[0]) - 1
wrapper = rfdetr.RFDETRSegMedium(
    num_classes=num_classes,
    resolution=RES,
    pretrain_weights=str(WEIGHTS),
    device=str(device),
)
core = wrapper.model.model.to(device).eval()
print("core loaded:", type(core).__name__, "num_classes:", num_classes)

dummy = torch.randn(BATCH, 3, RES, RES, device=device)

# seg head antialiased bicubic has no ONNX symbol; drop antialias for the trace
import torch.nn.functional as F  # noqa: E402

_orig_interpolate = F.interpolate


def _no_aa_interpolate(input, size=None, scale_factor=None, mode="nearest", **kw):
    kw.pop("antialias", None)
    return _orig_interpolate(input, size, scale_factor, mode, **kw)


F.interpolate = _no_aa_interpolate
with torch.inference_mode():
    torch.onnx.export(
        core,
        (dummy,),
        str(ONNX_PATH),
        opset_version=19,
        input_names=["input"],
        output_names=["dets", "labels", "masks"],
        dynamic_axes=None,
        do_constant_folding=True,
        dynamo=False,
    )
F.interpolate = _orig_interpolate
print("exported:", ONNX_PATH, f"{ONNX_PATH.stat().st_size / 1e6:.0f} MB")
