"""
adapter.py -- the ONLY place fusion reads Role 2's AlignedFrameSet.

Contract (docs/data-contract.md 6.2): fusion may condition on
frames[].ransacInlierRatio ONLY. The classifier's 'alignmentConfidence' is
never read here, on purpose.
"""
import numpy as np

RATIO_KEY = "ransacInlierRatio"      # the only confidence fusion may use


def alignedset_to_arrays(aligned_result):
    """AlignedFrameSet dict -> (frames NHWC float32 0-1, ratios (N,) float32).

    Skips frames with no image and frames that did not actually get warped.
    Channel order is whatever Role 2 loaded (cv2.imread -> BGR); unchanged here. Raises on anything that looks wrong
    instead of silently feeding fusion bad numbers.
    """
    if aligned_result is None:
        raise ValueError("aligned_result is None (alignment failed upstream)")
    imgs, ratios = [], []
    ref = aligned_result.get("referenceFrameIndex")
    for fr in aligned_result["frames"]:
        img = fr.get("image")
        if img is None:
            continue
        # Role 2 returns the UNWARPED image (with aligned=False) when alignment
        # fails, sometimes with a nonzero inlier ratio. Only the reference frame
        # and frames flagged aligned=True are real aligned pixels.
        if fr.get("frameIndex") != ref and fr.get("aligned") is not True:
            continue
        if RATIO_KEY not in fr:
            raise KeyError(f"frame {fr.get('frameIndex')} has no '{RATIO_KEY}'")
        r = float(fr[RATIO_KEY])
        if not (0.0 <= r <= 1.0) or np.isnan(r):
            raise ValueError(f"{RATIO_KEY}={r} outside 0..1 on frame {fr.get('frameIndex')}")
        img = np.asarray(img)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            if img.max() > 1.0 + 1e-3:
                raise ValueError("float image with values >1: expected 0..1 or uint8")
        imgs.append(img); ratios.append(r)
    if not imgs:
        raise ValueError("no usable frames in aligned set")
    shapes = {i.shape for i in imgs}
    if len(shapes) != 1:
        raise ValueError(f"aligned frames differ in shape: {shapes}")
    return np.stack(imgs), np.asarray(ratios, dtype=np.float32)