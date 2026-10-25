"""
module.py -- FusionModule entry point: AlignedFrameSet in, FusedFrame out.

    fused, info = fuse(aligned_result, strategy="confidence", pipeline_mode="fusion_multi")

pipeline_mode is Role 1's flag (proposal 4.3):
  * fusion_multi  -> merge the aligned frames with the chosen strategy
  * fusion_single -> pass the reference (sharpest) frame through untouched
Frames are read ONLY through fusion.adapter, so only ransacInlierRatio is ever used.
"""
import numpy as np

from fusion.adapter import alignedset_to_arrays
from fusion.strategies import naivemeanfusion, trimmedmeanfusion, confidenceweightedfusion

STRATEGIES = ("naive", "trimmed", "confidence", "learned")
MODES = ("fusion_multi", "fusion_single")


def fuse_arrays(frames, ratios, strategy, trimfrac=0.4, floor=0.0, model=None):
    """Fuse already-converted arrays (frames NHWC float32 0-1, ratios (N,)). Returns (fused, info)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy '{strategy}', pick one of {STRATEGIES}")
    keep = np.asarray(ratios) >= floor
    keep[int(np.argmax(ratios))] = True          # reference frame never dropped
    frames, ratios = frames[keep], np.asarray(ratios, dtype=np.float32)[keep]
    used = strategy
    if strategy == "naive" or len(frames) == 1:
        fused = naivemeanfusion(frames)
        used = "naive" if len(frames) > 1 else "single"
    elif strategy == "trimmed":
        if len(frames) >= 3:
            fused = trimmedmeanfusion(frames, trimfrac)
        else:                                    # nothing to trim with 2 frames
            fused, used = naivemeanfusion(frames), "naive"
    elif strategy == "confidence":
        fused = confidenceweightedfusion(frames, ratios)
    else:
        if model is None:
            raise ValueError("strategy 'learned' needs model= (see learned_fusion/infer.py load_model)")
        from learned_fusion.infer import learnedfusion
        fused = learnedfusion(frames, ratios, model)
    return np.clip(fused, 0.0, 1.0).astype(np.float32), {
        "strategy_requested": strategy, "strategy_used": used,
        "frames_fused": int(len(frames)), "frames_dropped_by_floor": int((~keep).sum())}


def fuse(aligned_result, strategy="confidence", pipeline_mode="fusion_multi", trimfrac=0.4, floor=0.0, model=None):
    """AlignedFrameSet dict -> (FusedFrame HWC float32 0-1, info dict)."""
    if pipeline_mode not in MODES:
        raise ValueError(f"unknown pipeline_mode '{pipeline_mode}', expected {MODES}")
    frames, ratios = alignedset_to_arrays(aligned_result)
    if pipeline_mode == "fusion_single":
        ref = int(np.argmax(ratios))             # reference frame carries ratio 1.0
        return frames[ref].copy(), {"strategy_requested": strategy, "strategy_used": "single",
                                    "frames_fused": 1, "frames_dropped_by_floor": 0, "pipeline_mode": pipeline_mode}
    fused, info = fuse_arrays(frames, ratios, strategy, trimfrac=trimfrac, floor=floor, model=model)
    info["pipeline_mode"] = pipeline_mode
    return fused, info
