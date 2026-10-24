import sys
import cv2
import os
import numpy as np


BURST_DIR = "burst"
REFERENCE_FRAME = "frame_11.jpeg"

# ORB configuration
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
orb = cv2.ORB_create(nfeatures=3000, fastThreshold=10)

# BFMatcher for ORB binary descriptors
matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)


# --------------------------------------------------
# Load reference frame
# --------------------------------------------------

reference_path = os.path.join(BURST_DIR, REFERENCE_FRAME)

reference = cv2.imread(reference_path)

if reference is None:
    raise RuntimeError(
        f"Could not load reference frame: {REFERENCE_FRAME}"
    )

reference_gray = cv2.cvtColor(
    reference,
    cv2.COLOR_BGR2GRAY
)


# --------------------------------------------------
# Detect ORB features in reference
# --------------------------------------------------

kp_ref, des_ref = orb.detectAndCompute(clahe.apply(reference_gray), None)

if des_ref is None:
    print(f"Reference frame: {REFERENCE_FRAME}")
    print("Reference keypoints: 0")
    for filename in sorted(os.listdir(BURST_DIR)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if filename == REFERENCE_FRAME:
            continue
        print(f"\n{filename}")
        print("Not enough keypoints/descriptors.")
    sys.exit(0)

print(f"Reference frame: {REFERENCE_FRAME}")
print(f"Reference keypoints: {len(kp_ref)}")


# --------------------------------------------------
# Process every other frame
# --------------------------------------------------

for filename in sorted(os.listdir(BURST_DIR)):

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    if filename == REFERENCE_FRAME:
        continue

    path = os.path.join(BURST_DIR, filename)

    image = cv2.imread(path)

    if image is None:
        print(f"\n{filename}: Could not load image.")
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ----------------------------------------------
    # ORB detection
    # ----------------------------------------------

    kp, des = orb.detectAndCompute(clahe.apply(gray), None)

    if des is None or len(kp) < 4:
        print(f"\n{filename}")
        print("Not enough keypoints/descriptors.")
        continue

    # ----------------------------------------------
    # Descriptor matching
    # ----------------------------------------------

    matches = matcher.match(
        des_ref,
        des
    )

    matches = sorted(
        matches,
        key=lambda m: m.distance
    )

    total_matches = len(matches)

    print(f"\n{filename}")
    print(f"Keypoints: {len(kp)}")
    print(f"Total matches: {total_matches}")

    # Homography requires at least 4 point correspondences
    if total_matches < 4:
        print("RANSAC: FAILED - fewer than 4 matches")
        print("Confidence: 0.0000")
        continue

    # ----------------------------------------------
    # Extract matched coordinates
    # ----------------------------------------------

    src_points = np.float32([
        kp_ref[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    dst_points = np.float32([
        kp[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    # ----------------------------------------------
    # RANSAC homography
    # ----------------------------------------------

    H, mask = cv2.findHomography(src_points, dst_points, cv2.USAC_MAGSAC, 3.0, maxIters=2000, confidence=0.995)

    # ----------------------------------------------
    # Check RANSAC result
    # ----------------------------------------------

    if H is None or mask is None:
        print("RANSAC: FAILED")
        print("Confidence: 0.0000")
        continue

    # mask contains:
    # 1 = inlier
    # 0 = outlier

    inlier_count = int(
        np.sum(mask.ravel() > 0)
    )

    # ----------------------------------------------
    # Prelude confidence score
    # ----------------------------------------------

    confidence = float(inlier_count / total_matches)

    print(f"RANSAC inliers: {inlier_count}")
    print(f"RANSAC confidence: {confidence:.4f}")
    print(f"RANSAC confidence (%): {confidence * 100:.2f}%")