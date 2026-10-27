# training/dataset.py
import logging
import math
import random
from pathlib import Path
from typing import List, Tuple

import keras
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("Dataset")

class SIDDMediumDataset(keras.utils.Sequence):
    def __init__(
        self,
        pairs: List[Tuple[str, Path, Path]],
        patch_size: int = 128,
        batch_size: int = 16,
        is_train: bool = True,
        inject_synthetic_noise: bool = False
    ):
        super().__init__()
        self.pairs = list(pairs)
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.is_train = is_train
        self.inject_synthetic_noise = inject_synthetic_noise
        if self.is_train:
            random.shuffle(self.pairs)

    def __len__(self) -> int:
        return math.ceil(len(self.pairs) / self.batch_size)

    def on_epoch_end(self):
        if self.is_train:
            random.shuffle(self.pairs)

    def _load_single_pair(self, idx: int) -> Tuple[str, np.ndarray, np.ndarray]:
        image_id, noisy_path, gt_path = self.pairs[idx]

        try:
            gt_img = Image.open(gt_path).convert("RGB")
            noisy_img = gt_img if self.inject_synthetic_noise else Image.open(noisy_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Failed to load image pair {image_id}: {e}")
            empty = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
            return image_id, empty, empty

        w, h = gt_img.size

        if w < self.patch_size or h < self.patch_size:
            new_w, new_h = max(w, self.patch_size), max(h, self.patch_size)
            gt_img = gt_img.resize((new_w, new_h))
            noisy_img = noisy_img.resize((new_w, new_h))
            w, h = gt_img.size

        if self.is_train:
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)
        else:
            x = (w - self.patch_size) // 2
            y = (h - self.patch_size) // 2

        box = (x, y, x + self.patch_size, y + self.patch_size)
        gt_crop = gt_img.crop(box)
        noisy_crop = noisy_img.crop(box)

        # Normalize to float32 [0.0, 1.0] in NHWC channel-last format
        gt_arr = np.asarray(gt_crop, dtype=np.float32) / 255.0
        noisy_arr = np.asarray(noisy_crop, dtype=np.float32) / 255.0

        if self.is_train:
            if random.random() > 0.5:
                gt_arr = np.fliplr(gt_arr)
                noisy_arr = np.fliplr(noisy_arr)
            if random.random() > 0.5:
                gt_arr = np.flipud(gt_arr)
                noisy_arr = np.flipud(noisy_arr)
            k_rot = random.randint(0, 3)
            if k_rot > 0:
                gt_arr = np.rot90(gt_arr, k_rot)
                noisy_arr = np.rot90(noisy_arr, k_rot)

        if self.inject_synthetic_noise:
            noise_std = np.random.uniform(30.0, 85.0) / 255.0
            synthetic_noise = np.random.normal(0.0, noise_std, size=gt_arr.shape).astype(np.float32)
            noisy_arr = np.clip(gt_arr + synthetic_noise, 0.0, 1.0)

        return image_id, np.ascontiguousarray(noisy_arr), np.ascontiguousarray(gt_arr)

    def __getitem__(self, batch_idx: int) -> Tuple[List[str], np.ndarray, np.ndarray]:
        start = batch_idx * self.batch_size
        end = min((batch_idx + 1) * self.batch_size, len(self.pairs))

        batch_ids, batch_noisy, batch_gt = [], [], []
        for idx in range(start, end):
            img_id, n_arr, g_arr = self._load_single_pair(idx)
            batch_ids.append(img_id)
            batch_noisy.append(n_arr)
            batch_gt.append(g_arr)

        return (
            batch_ids,
            np.stack(batch_noisy, axis=0),
            np.stack(batch_gt, axis=0)
        )