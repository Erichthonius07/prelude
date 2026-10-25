"""
dataset.py -- loads the .npz bursts written by build_dataset.py for training.

Each npz: frames (N,H,W,3) uint8 BGR, ratios (N,) ransacInlierRatio, target (H,W,3) uint8.
Warp borders are cropped by `inset` px before anything else (aligned frames have
empty edges there). Bursts are padded to MAX_FRAMES with a mask.
"""
from pathlib import Path
import numpy as np

MAX_FRAMES = 8
FORBIDDEN_SPLITS = ("test", "heldout", "held_out")   # Role 1's test set never trains this model


def list_bursts(root, split):
    if split.lower() in FORBIDDEN_SPLITS:
        raise ValueError(f"split '{split}' looks like the held-out test set; the learned fusion model must never train on it")
    files = sorted((Path(root) / split).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz bursts in {Path(root) / split} (run build_dataset.py first)")
    return files


def load_burst(path, inset=48):
    d = np.load(path)
    frames, ratios, target = d["frames"], d["ratios"].astype(np.float32), d["target"]
    if inset:
        frames = frames[:, inset:-inset, inset:-inset]
        target = target[inset:-inset, inset:-inset]
    if len(frames) > MAX_FRAMES:   # keep the most confident frames
        order = np.argsort(-ratios)[:MAX_FRAMES]
        frames, ratios = frames[order], ratios[order]
    return frames.astype(np.float32) / 255.0, ratios, target.astype(np.float32) / 255.0


def to_arrays(frames, ratios, target=None):
    """numpy burst -> padded arrays (frames (8,H,W,3), ratios (8,), mask (8,), target (H,W,3))."""
    n, h, w, _ = frames.shape
    f = np.zeros((MAX_FRAMES, h, w, 3), np.float32)
    f[:n] = frames
    r = np.zeros(MAX_FRAMES, np.float32); r[:n] = ratios
    m = np.zeros(MAX_FRAMES, bool); m[:n] = True
    out = [f, r, m]
    if target is not None:
        out.append(np.ascontiguousarray(target, dtype=np.float32))
    return out


class PatchDataset:
    """Random patches with flip / 90-degree rotation augmentation; whole bursts held in memory."""

    def __init__(self, files, patch=128, patches_per_burst=4, inset=48, augment=True, seed=0):
        self.bursts = [load_burst(f, inset) for f in files]
        self.patch, self.ppb, self.augment = patch, patches_per_burst, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.bursts) * self.ppb

    def __getitem__(self, i):
        frames, ratios, target = self.bursts[i // self.ppb]
        h, w = target.shape[:2]
        p = min(self.patch, h, w)
        y = int(self.rng.integers(0, h - p + 1)); x = int(self.rng.integers(0, w - p + 1))
        fr, tg = frames[:, y:y + p, x:x + p], target[y:y + p, x:x + p]
        if self.augment:
            k = int(self.rng.integers(0, 4))
            fr, tg = np.rot90(fr, k, axes=(1, 2)), np.rot90(tg, k, axes=(0, 1))
            if self.rng.random() < 0.5:
                fr, tg = fr[:, :, ::-1], tg[:, ::-1]
        return to_arrays(np.ascontiguousarray(fr), ratios, np.ascontiguousarray(tg))

    def batches(self, batch):
        """One shuffled epoch of stacked (frames, ratios, mask, target) batches."""
        order = self.rng.permutation(len(self))
        for k in range(0, len(order), batch):
            items = [self[i] for i in order[k:k + batch]]
            yield [np.stack(col) for col in zip(*items)]
