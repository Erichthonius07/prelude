"""infer.py -- load a trained checkpoint and fuse a numpy burst with it (strategy 4)."""
import json
from pathlib import Path
import numpy as np

from learned_fusion.model import FusionWeightNet
from learned_fusion.dataset import MAX_FRAMES, to_arrays


def load_model(checkpoint):
    """checkpoint = <name>.weights.h5 written by train.py; its config sits next to it as <name>.json."""
    meta = json.loads(Path(str(checkpoint).replace(".weights.h5", ".json")).read_text())
    model = FusionWeightNet(**meta["config"])
    f, r, m = to_arrays(np.zeros((1, 16, 16, 3), np.float32), np.ones(1, np.float32))
    model(f[None], r[None], m[None])                     # build the layers before loading weights
    model.load_weights(str(checkpoint))
    return model


def learnedfusion(frames, ratios, model, return_weights=False):
    """frames (N,H,W,3) float32 0-1, ratios (N,) ransacInlierRatio -> fused (H,W,3) float32."""
    ratios = np.asarray(ratios, dtype=np.float32)
    if len(frames) > MAX_FRAMES:
        order = np.argsort(-ratios)[:MAX_FRAMES]
        frames, ratios = frames[order], ratios[order]
    f, r, m = to_arrays(np.asarray(frames, np.float32), ratios)
    fused, weights = model(f[None], r[None], m[None])
    out = fused[0].numpy().clip(0, 1).astype(np.float32)
    if return_weights:
        return out, weights[0, :len(frames)].numpy()
    return out
