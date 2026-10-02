import os
import cv2
import numpy as np


def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def align_burst(burst_path):
    image_files = sorted([
        f for f in os.listdir(burst_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if len(image_files) < 2:
        return None

    images = {}

    for filename in image_files:
        image_path = os.path.join(burst_path, filename)
        image = cv2.imread(image_path)

        if image is not None:
            images[filename] = image

    if len(images) < 2:
        return None

    sharpness_scores = {
        filename: calculate_sharpness(image)
        for filename, image in images.items()
    }

    reference_frame = max(
        sharpness_scores,
        key=sharpness_scores.get
    )

    reference = images[reference_frame]

    orb = cv2.ORB_create(nfeatures=2000)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    reference_keypoints, reference_descriptors = orb.detectAndCompute(
        reference_gray,
        None
    )

    if reference_descriptors is None:
        return None

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    frames = []

    reference_index = image_files.index(reference_frame)

    for frame_index, filename in enumerate(image_files):

        image = images.get(filename)

        if image is None:
            frames.append({
                "frameIndex": frame_index,
                "image": None,
                "homography": None,
                "ransacInlierRatio": 0.0,
                "aligned": False
            })
            continue

        if filename == reference_frame:
            frames.append({
                "frameIndex": frame_index,
                "image": image,
                "homography": np.eye(3, dtype=np.float32),
                "ransacInlierRatio": 1.0,
                "aligned": True
            })
            continue

        current_gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        current_keypoints, current_descriptors = orb.detectAndCompute(
            current_gray,
            None
        )

        if current_descriptors is None:
            frames.append({
                "frameIndex": frame_index,
                "image": image,
                "homography": None,
                "ransacInlierRatio": 0.0,
                "aligned": False
            })
            continue

        matches = matcher.match(
            reference_descriptors,
            current_descriptors
        )

        total_matches = len(matches)

        if total_matches < 4:
            frames.append({
                "frameIndex": frame_index,
                "image": image,
                "homography": None,
                "ransacInlierRatio": 0.0,
                "aligned": False
            })
            continue

        matches = sorted(
            matches,
            key=lambda match: match.distance
        )

        src_points = np.float32([
            reference_keypoints[m.queryIdx].pt
            for m in matches
        ]).reshape(-1, 1, 2)

        dst_points = np.float32([
            current_keypoints[m.trainIdx].pt
            for m in matches
        ]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_points,
            dst_points,
            cv2.RANSAC,
            5.0,
            maxIters=2000,
            confidence=0.995
        )

        if H is None or mask is None:
            frames.append({
                "frameIndex": frame_index,
                "image": image,
                "homography": None,
                "ransacInlierRatio": 0.0,
                "aligned": False
            })
            continue

        inliers = int(np.sum(mask.ravel() > 0))
        inlier_ratio = inliers / total_matches

        try:
            H_inv = np.linalg.inv(H)

            warped_image = cv2.warpPerspective(
                image,
                H_inv,
                (reference.shape[1], reference.shape[0])
            )

        except np.linalg.LinAlgError:
            frames.append({
                "frameIndex": frame_index,
                "image": image,
                "homography": None,
                "ransacInlierRatio": inlier_ratio,
                "aligned": False
            })
            continue

        frames.append({
            "frameIndex": frame_index,
            "image": warped_image,
            "_originalImage": image,
            "homography": H,
            "ransacInlierRatio": inlier_ratio,
            "aligned": True
        })

    return {
        "burstId": os.path.basename(os.path.normpath(burst_path)),
        "referenceFrameIndex": reference_index,
        "frames": frames
    }