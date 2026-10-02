import cv2
import numpy as np


def extract_features(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    laplacian_variance = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    orb = cv2.ORB_create(
        nfeatures=2000
    )

    keypoints, _ = orb.detectAndCompute(
        gray,
        None
    )

    keypoint_count = len(keypoints)

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitude = np.sqrt(
        sobel_x ** 2 +
        sobel_y ** 2
    )

    magnitude = np.clip(
        magnitude,
        0,
        1024
    )

    histogram, _ = np.histogram(
        magnitude,
        bins=16,
        range=(0, 1024)
    )

    histogram = histogram.astype(float)

    histogram /= (
        histogram.sum() + 1e-8
    )

    features = [
        laplacian_variance,
        keypoint_count,
        magnitude.mean(),
        magnitude.std()
    ]

    features.extend(
        histogram.tolist()
    )

    return np.array(features)