"""
alignment_experiment.py

Compares Role 2 alignment settings on synthetic bursts with known motion.

The baseline replicates:
ORB -> BFMatcher crossCheck -> findHomography -> inverse homography
-> warp direction from current frame to reference.

Includes a Lowe ratio 0.65 candidate matching Role 2's new configuration.

Run from repository root:
python fusion-postprocess-demo/learned_fusion/alignment_experiment.py \
    --bursts data/synthetic_bursts_clean --limit 60
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


T = np.array([
    [1.0, 0.0, 64.0],
    [0.0, 1.0, 64.0],
    [0.0, 0.0, 1.0],
])

TI = np.linalg.inv(T)

PTS = np.array([
    [50.0, 50.0, 1.0],
    [450.0, 60.0, 1.0],
    [60.0, 450.0, 1.0],
    [450.0, 450.0, 1.0],
]).T


# name, ORB features, RANSAC threshold, pre-blur sigma, Lowe ratio
# None means BFMatcher crossCheck, matching the original Role 2 baseline.
CONFIGS = [
    ("baseline (Role 2 as written)", 2000, 5.0, 0.0, None),

    # Candidate matching the newer Role 2 alignment settings.
    ("Lowe ratio 0.65", 2000, 5.0, 0.0, 0.65),

    ("RANSAC 3 px", 2000, 3.0, 0.0, None),
    ("RANSAC 2 px", 2000, 2.0, 0.0, None),
    ("RANSAC 1 px", 2000, 1.0, 0.0, None),

    ("pre-blur 1.5", 2000, 5.0, 1.5, None),
    ("pre-blur 1.5 + RANSAC 2 px", 2000, 2.0, 1.5, None),
    ("pre-blur 2.5 + RANSAC 2 px", 2000, 2.0, 2.5, None),
    ("5000 feat + blur 1.5 + 2 px", 5000, 2.0, 1.5, None),
    ("ratio test .75 + blur 1.5 + 2 px", 2000, 2.0, 1.5, 0.75),
]


def corner_err(matrix, true_matrix):
    """Maximum absolute coordinate error over four test corners."""
    predicted = matrix @ PTS
    predicted = predicted[:2] / predicted[2]

    actual = true_matrix @ PTS
    actual = actual[:2] / actual[2]

    return float(np.abs(predicted - actual).max())


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def estimate(ref_gray, cur_gray, nfeatures, threshold, blur, lowe_ratio):
    """Estimate a reference-to-current homography."""
    if blur > 0:
        ref_gray = cv2.GaussianBlur(
            ref_gray, (0, 0), blur
        )
        cur_gray = cv2.GaussianBlur(
            cur_gray, (0, 0), blur
        )

    orb = cv2.ORB_create(nfeatures=nfeatures)

    kp_ref, desc_ref = orb.detectAndCompute(ref_gray, None)
    kp_cur, desc_cur = orb.detectAndCompute(cur_gray, None)

    if desc_ref is None or desc_cur is None:
        return None, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    if lowe_ratio is not None:
        pairs = matcher.knnMatch(
            desc_ref,
            desc_cur,
            k=2,
        )

        matches = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < lowe_ratio * second.distance
        ]
    else:
        matches = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=True,
        ).match(desc_ref, desc_cur)

    if len(matches) < 4:
        return None, 0.0

    src = np.float32([
        kp_ref[match.queryIdx].pt
        for match in matches
    ]).reshape(-1, 1, 2)

    dst = np.float32([
        kp_cur[match.trainIdx].pt
        for match in matches
    ]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        threshold,
        maxIters=2000,
        confidence=0.995,
    )

    if homography is None or mask is None:
        return None, 0.0

    inlier_ratio = float(mask.sum()) / len(matches)

    return homography, inlier_ratio


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--bursts",
        required=True,
        type=Path,
        help="Synthetic burst dataset folder containing manifest.json",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="Maximum number of bursts to evaluate",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    manifest_path = args.bursts / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    items = list(manifest["bursts"].items())

    if not items:
        raise RuntimeError("No bursts found in manifest.json")

    step = max(1, len(items) // args.limit)
    items = items[::step][:args.limit]

    data = []

    print("Loading synthetic bursts...")

    for burst_id, burst_info in items:
        burst_dir = (
            args.bursts
            / burst_info["split"]
            / burst_id
        )

        if not burst_dir.is_dir():
            print(f"Skipping missing burst directory: {burst_dir}")
            continue

        files = sorted(
            filename
            for filename in os.listdir(burst_dir)
            if filename.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        )

        if len(files) < 2:
            print(f"Skipping burst with fewer than 2 images: {burst_id}")
            continue

        images = [
            cv2.imread(str(burst_dir / filename))
            for filename in files
        ]

        if any(image is None for image in images):
            print(f"Skipping burst with unreadable images: {burst_id}")
            continue

        gray_images = [gray(image) for image in images]

        # This assumes the variance-of-Laplacian calculation matches
        # Role 2's calculate_sharpness() implementation.
        sharpness = [
            cv2.Laplacian(image, cv2.CV_64F).var()
            for image in gray_images
        ]

        reference_index = int(np.argmax(sharpness))

        homographies = [
            np.asarray(h, dtype=np.float64)
            for h in burst_info["homographies"]
        ]

        if len(homographies) != len(gray_images):
            print(
                f"Skipping {burst_id}: "
                f"{len(gray_images)} images but "
                f"{len(homographies)} homographies"
            )
            continue

        data.append((
            burst_id,
            gray_images,
            homographies,
            reference_index,
        ))

    if not data:
        raise RuntimeError(
            "No usable bursts loaded. Check the dataset path and manifest."
        )

    total_pairs = sum(
        len(images) - 1
        for _, images, _, _ in data
    )

    print(
        f"\nLoaded {len(data)} bursts, "
        f"{total_pairs} frame pairs\n"
    )

    header = (
        f"{'setting':36s} | fail% | median px | "
        f"<2px | worse than nothing | ratio>=.5 correct"
    )

    print(header)
    print("-" * len(header))

    for name, nfeatures, threshold, blur, lowe_ratio in CONFIGS:
        errors = []
        no_alignment_errors = []
        ratios = []
        failed = 0

        for burst_id, images, homographies, ref_idx in data:
            ref_gray = images[ref_idx]

            for frame_index, cur_gray in enumerate(images):
                if frame_index == ref_idx:
                    continue

                # Ground-truth transform from the current frame to reference.
                true_matrix = (
                    TI
                    @ homographies[ref_idx]
                    @ np.linalg.inv(homographies[frame_index])
                    @ T
                )

                identity_error = corner_err(
                    np.eye(3),
                    true_matrix,
                )

                no_alignment_errors.append(identity_error)

                estimated_h, ratio = estimate(
                    ref_gray,
                    cur_gray,
                    nfeatures,
                    threshold,
                    blur,
                    lowe_ratio,
                )

                # Failed estimation is treated as leaving the frame unaligned.
                if estimated_h is None:
                    failed += 1
                    errors.append(identity_error)
                    ratios.append(0.0)
                    continue

                try:
                    estimated_inverse = np.linalg.inv(estimated_h)
                    error = corner_err(
                        estimated_inverse,
                        true_matrix,
                    )
                except np.linalg.LinAlgError:
                    failed += 1
                    errors.append(identity_error)
                    ratios.append(0.0)
                    continue

                errors.append(error)
                ratios.append(ratio)

        e = np.asarray(errors, dtype=np.float64)
        no_align = np.asarray(
            no_alignment_errors,
            dtype=np.float64,
        )
        r = np.asarray(ratios, dtype=np.float64)

        high_ratio = r >= 0.5

        if np.any(high_ratio):
            ratio_correct = (
                f"{np.mean(e[high_ratio] < 2):4.0%} of "
                f"{high_ratio.mean():4.0%}"
            )
        else:
            ratio_correct = "n/a"

        print(
            f"{name:36s} | "
            f"{failed / len(e):4.0%}  | "
            f"{np.median(e):9.2f} | "
            f"{np.mean(e < 2):4.0%} | "
            f"{np.mean(e > no_align):17.0%} | "
            f"{ratio_correct}"
        )

    print("\nInterpretation:")
    print("- Lower median px is better.")
    print("- Higher <2px is better.")
    print("- Lower fail% is better.")
    print("- 'Worse than nothing' should be lower.")
    print(
        "- Baseline should be checked against diagnose_alignment.py "
        "on the same bursts to verify replica fidelity."
    )
    print(
        "- This measures synthetic geometric alignment error, "
        "not final denoising PSNR/SSIM."
    )


if __name__ == "__main__":
    main()
