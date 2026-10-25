import numpy as np
import pytest
from fusion.strategies import confidenceweightedfusion, trimmedmeanfusion, naivemeanfusion
from fusion.module import fuse, fuse_arrays


def _alignedset(frames, ratios, aligned=None):
    aligned = aligned or [True] * len(frames)
    return {"referenceFrameIndex": 0, "frames": [
        {"frameIndex": i, "image": f, "ransacInlierRatio": r, "aligned": a, "alignmentConfidence": 0.0}
        for i, (f, r, a) in enumerate(zip(frames, ratios, aligned))]}


def test_confidence_equal_ratios_is_naive():
    burst = np.random.default_rng(0).uniform(0, 1, (5, 8, 8, 3)).astype(np.float32)
    assert np.allclose(confidenceweightedfusion(burst, [0.7] * 5), naivemeanfusion(burst), atol=1e-6)


def test_confidence_downweights_bad_frame():
    rng = np.random.default_rng(1)
    scene = rng.uniform(0.3, 0.7, (32, 32, 3)).astype(np.float32)
    burst = np.stack([scene + rng.normal(0, 0.03, scene.shape) for _ in range(5)]).astype(np.float32)
    burst[4] = np.roll(burst[4], 6, axis=1)                  # misaligned frame
    ratios = [1.0, 0.8, 0.8, 0.8, 0.1]
    err_c = np.abs(confidenceweightedfusion(burst, ratios) - scene).mean()
    err_n = np.abs(naivemeanfusion(burst) - scene).mean()
    assert err_c < err_n


def test_confidence_floor_and_errors():
    burst = np.ones((3, 4, 4, 3), np.float32) * np.array([0.2, 0.5, 0.9])[:, None, None, None].astype(np.float32)
    out = confidenceweightedfusion(burst, [1.0, 0.5, 0.2], floor=0.3)
    assert np.allclose(out, (0.2 * 1.0 + 0.5 * 0.5) / 1.5)
    with pytest.raises(ValueError):
        confidenceweightedfusion(burst, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        confidenceweightedfusion(burst, [1.0, 0.5])


def test_trimmed_needs_two_left():
    burst = np.zeros((3, 4, 4, 3), np.float32)
    with pytest.raises(ValueError):
        trimmedmeanfusion(burst, trimfrac=0.7)               # trims 1 each side -> 1 left


def test_fuse_single_mode_returns_reference():
    rng = np.random.default_rng(2)
    frames = [rng.uniform(0, 1, (8, 8, 3)).astype(np.float32) for _ in range(4)]
    out, info = fuse(_alignedset(frames, [1.0, 0.6, 0.6, 0.6]), "confidence", "fusion_single")
    assert np.allclose(out, frames[0]) and info["strategy_used"] == "single"


def test_fuse_multi_all_strategies_and_unaligned_skipped():
    rng = np.random.default_rng(3)
    frames = [rng.uniform(0, 1, (8, 8, 3)).astype(np.float32) for _ in range(5)]
    aset = _alignedset(frames, [1.0, 0.7, 0.6, 0.5, 0.9], aligned=[True, True, True, True, False])
    for s in ("naive", "trimmed", "confidence"):
        out, info = fuse(aset, s)
        assert out.shape == (8, 8, 3) and info["frames_fused"] == 4
    with pytest.raises(ValueError):
        fuse(aset, "learned")                                # no model given
    with pytest.raises(ValueError):
        fuse(aset, "naive", "fusion_bogus")


def test_fuse_arrays_floor_keeps_reference():
    frames = np.random.default_rng(4).uniform(0, 1, (3, 4, 4, 3)).astype(np.float32)
    out, info = fuse_arrays(frames, np.array([1.0, 0.1, 0.1]), "trimmed", floor=0.5)
    assert np.allclose(out, frames[0]) and info["frames_dropped_by_floor"] == 2
