import cv2
import pickle
import numpy as np
import sys

MODEL_PATH = r"C:\Prelude\alignment_classifier.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian_variance = cv2.Laplacian(
        gray, cv2.CV_64F
    ).var()

    orb = cv2.ORB_create(nfeatures=2000)
    keypoints, _ = orb.detectAndCompute(gray, None)
    orb_keypoint_count = len(keypoints)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(
        sobel_x ** 2 + sobel_y ** 2
    )

    gradient_mean = gradient_magnitude.mean()
    gradient_std = gradient_magnitude.std()

    gradient_clipped = np.clip(
        gradient_magnitude, 0, 1024
    )

    histogram, _ = np.histogram(
        gradient_clipped,
        bins=16,
        range=(0, 1024)
    )

    histogram = histogram.astype(float)
    histogram /= histogram.sum() + 1e-8

    features = [
        laplacian_variance,
        orb_keypoint_count,
        gradient_mean,
        gradient_std
    ]

    features.extend(histogram.tolist())

    return np.array(features).reshape(1, -1)


if len(sys.argv) < 2:
    print("Usage:")
    print("python classifier_predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not load image")
    sys.exit(1)

features = extract_features(image)

good_probability = model.predict_proba(features)[0][1]

prediction = 1 if good_probability >= 0.5 else 0

print("===== ALIGNMENT CLASSIFIER =====")
print(f"Image           : {image_path}")
print(f"Good probability: {good_probability:.4f}")
print(f"Prediction      : {'GOOD' if prediction == 1 else 'BAD'}")

if prediction == 1:
    print("Action          : RUN RANSAC")
else:
    print("Action          : EXCLUDE BEFORE RANSAC")