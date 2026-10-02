# training/dataset.py
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

logger = logging.getLogger("SIDDDataset")

class SIDDMediumDataset(Dataset):
    def __init__(
        self, 
        pairs: List[Tuple[str, Path, Path]], 
        patch_size: int = 128, 
        is_train: bool = True
    ):
        """
        pairs: List of tuples containing (imageId, noisy_image_path, ground_truth_path).
        patch_size: The square size to crop for training (e.g., 128x128).
        is_train: If True, applies random cropping. If False, applies a deterministic center crop.
        """
        self.pairs = pairs
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        image_id, noisy_path, gt_path = self.pairs[idx]
        
        try:
            # Enforce strict RGB conversion to prevent channel mismatch errors (e.g., RGBA or Grayscale)
            noisy_img = Image.open(noisy_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Failed to load image pair {image_id}: {e}")
            # Fallback to a zero-tensor if a file is corrupted to prevent the whole epoch from crashing
            empty_tensor = torch.zeros((3, self.patch_size, self.patch_size), dtype=torch.float32)
            return image_id, empty_tensor, empty_tensor

        w, h = noisy_img.size

        # Ensure the image is large enough for the requested patch size
        if w < self.patch_size or h < self.patch_size:
            noisy_img = noisy_img.resize((max(w, self.patch_size), max(h, self.patch_size)))
            gt_img = gt_img.resize((max(w, self.patch_size), max(h, self.patch_size)))
            w, h = noisy_img.size

        if self.is_train:
            # Random crop for training (acts as data augmentation)
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)
        else:
            # Deterministic center crop for validation to ensure consistent scoring
            x = (w - self.patch_size) // 2
            y = (h - self.patch_size) // 2

        # Crop both images using the EXACT same coordinates
        box = (x, y, x + self.patch_size, y + self.patch_size)
        noisy_crop = noisy_img.crop(box)
        gt_crop = gt_img.crop(box)

        # Convert to NumPy, then to PyTorch tensors with shape [Channels, Height, Width]
        noisy_tensor = torch.from_numpy(np.array(noisy_crop)).permute(2, 0, 1).float()
        gt_tensor = torch.from_numpy(np.array(gt_crop)).permute(2, 0, 1).float()

        # Normalize pixel values from [0, 255] to [0.0, 1.0] for neural network processing
        noisy_tensor /= 255.0
        gt_tensor /= 255.0

        return image_id, noisy_tensor, gt_tensor