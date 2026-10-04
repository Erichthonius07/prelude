#!/usr/bin/env python3
"""sidd_extract.py — extract SIDD validation block pairs for offline evaluation.

Source: the official SIDD download page (URL below), MATLAB v5 .mat arrays
[#images, #blocks, 256, 256, 3] uint8. Split BY IMAGE (scene), not by block:

  - calibration blocks: images 0..19  (<= 100 blocks, seeded choice)  -> calibration/
  - evaluation pairs:   images 20..39 (<=  50 blocks, seeded choice)  -> pairs/{input,ground_truth}/

The two image-index sets are disjoint by construction (asserted). Blocks are
stored as float32 in [0, 1] (uint8 / 255, sRGB domain — rule C3: this /255 IS
the declared sRGB decoding, nothing else). A provenance sidecar records source
URLs, file SHA-256s, the seed, and the chosen indices. No held-out data is
touched: this is the PUBLIC SIDD validation split (label: VALIDATION / srgb /
not held-out / not final).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SOURCE_PAGE = "http://130.63.97.225/share/download_benchmark.html"
NOISY_URL = "http://130.63.97.225/share/SIDD_Blocks/ValidationNoisyBlocksSrgb.mat"
GT_URL = "http://130.63.97.225/share/SIDD_Blocks/ValidationGtBlocksSrgb.mat"
SEED = 42
CALIBRATION_IMAGES = range(0, 20)   # images 0..19
EVALUATION_IMAGES = range(20, 40)   # images 20..39
MAX_CALIBRATION_BLOCKS = 100
MAX_EVALUATION_BLOCKS = 50


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract SIDD validation block pairs")
    parser.add_argument("--data-dir", required=True,
                        help="directory containing the two Validation*BlocksSrgb.mat files")
    parser.add_argument("--out-dir", required=True,
                        help="output root (calibration/, pairs/, provenance JSON)")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    noisy_path = data_dir / "ValidationNoisyBlocksSrgb.mat"
    gt_path = data_dir / "ValidationGtBlocksSrgb.mat"
    for p in (noisy_path, gt_path):
        if not p.is_file():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    noisy = sio.loadmat(noisy_path)["ValidationNoisyBlocksSrgb"]
    gt = sio.loadmat(gt_path)["ValidationGtBlocksSrgb"]
    if noisy.shape != gt.shape:
        print(f"ERROR: shape mismatch {noisy.shape} vs {gt.shape}", file=sys.stderr)
        return 2
    n_images, n_blocks = noisy.shape[0], noisy.shape[1]
    print(f"loaded noisy {noisy.shape} {noisy.dtype}, gt {gt.shape} {gt.dtype}")

    calib_images = list(CALIBRATION_IMAGES)
    eval_images = list(EVALUATION_IMAGES)
    if set(calib_images) & set(eval_images):
        print("ERROR: image split is not disjoint", file=sys.stderr)
        return 2
    if max(calib_images) >= n_images or max(eval_images) >= n_images:
        print(f"ERROR: image indices exceed {n_images} images", file=sys.stderr)
        return 2

    rng = np.random.default_rng(SEED)
    calib_pairs = [(i, int(b)) for i in calib_images for b in range(n_blocks)]
    eval_pairs = [(i, int(b)) for i in eval_images for b in range(n_blocks)]
    calib_chosen = sorted(rng.choice(len(calib_pairs), size=min(MAX_CALIBRATION_BLOCKS, len(calib_pairs)),
                                     replace=False).tolist())
    eval_chosen = sorted(rng.choice(len(eval_pairs), size=min(MAX_EVALUATION_BLOCKS, len(eval_pairs)),
                                    replace=False).tolist())
    calib_blocks = [calib_pairs[j] for j in calib_chosen]
    eval_blocks = [eval_pairs[j] for j in eval_chosen]

    # Disjointness proof (G2): same (image, block) never appears in both sets.
    assert not (set(calib_blocks) & set(eval_blocks)), "calibration/evaluation overlap"

    calib_dir = out_dir / "calibration"
    pair_in = out_dir / "pairs" / "input"
    pair_gt = out_dir / "pairs" / "ground_truth"
    for d in (calib_dir, pair_in, pair_gt):
        d.mkdir(parents=True, exist_ok=True)

    def to_f32(block: np.ndarray) -> np.ndarray:
        return (block.astype(np.float32) / np.float32(255.0)).copy()

    for i, b in calib_blocks:
        np.save(calib_dir / f"img{i:02d}_blk{b:02d}.npy", to_f32(noisy[i, b]))
    baseline = []
    for i, b in eval_blocks:
        np.save(pair_in / f"img{i:02d}_blk{b:02d}.npy", to_f32(noisy[i, b]))
        gt_arr = to_f32(gt[i, b])
        np.save(pair_gt / f"img{i:02d}_blk{b:02d}.npy", gt_arr)
        noisy_arr = to_f32(noisy[i, b])
        baseline.append({
            "blockId": f"img{i:02d}_blk{b:02d}",
            "psnr": float(peak_signal_noise_ratio(gt_arr, noisy_arr, data_range=1.0)),
            "ssim": float(structural_similarity(gt_arr, noisy_arr, data_range=1.0,
                                                channel_axis=2)),
        })

    provenance = {
        "source_page": SOURCE_PAGE,
        "noisy_url": NOISY_URL,
        "gt_url": GT_URL,
        "noisy_sha256": sha256_of(noisy_path),
        "gt_sha256": sha256_of(gt_path),
        "seed": SEED,
        "calibration_image_indices": calib_images,
        "evaluation_image_indices": eval_images,
        "calibration_block_indices": calib_blocks,
        "evaluation_block_indices": eval_blocks,
        "domain": "srgb (uint8 / 255, no other conversion — rule C3)",
        "label": "VALIDATION / srgb / not held-out / not final",
        "scipy_version": __import__("scipy").__version__,
        "noisy_vs_gt_baseline": baseline,
        "baseline_mean_psnr": float(np.mean([r["psnr"] for r in baseline])),
        "baseline_mean_ssim": float(np.mean([r["ssim"] for r in baseline])),
    }
    sidecar = out_dir / "extraction_provenance.json"
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"calibration blocks: {len(calib_blocks)} -> {calib_dir}")
    print(f"evaluation pairs:   {len(eval_blocks)} -> {pair_in.parent}")
    print(f"baseline noisy-vs-GT: mean PSNR {provenance['baseline_mean_psnr']:.3f} dB, "
          f"mean SSIM {provenance['baseline_mean_ssim']:.5f}")
    print(f"provenance: {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
