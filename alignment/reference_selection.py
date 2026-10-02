import cv2
import os

BURST_DIR = "burst"


def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


scores = []

for filename in os.listdir(BURST_DIR):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(BURST_DIR, filename)

    image = cv2.imread(path)

    if image is None:
        print(f"Could not load: {filename}")
        continue

    score = variance_of_laplacian(image)

    scores.append((filename, score))


if not scores:
    raise RuntimeError("No valid images found in burst folder.")


scores.sort(key=lambda x: x[1], reverse=True)

print("\nSharpness scores:")

for filename, score in scores:
    print(f"{filename}: {score:.2f}")


reference_frame, reference_score = scores[0]

print("\nReference frame:")
print(reference_frame)
print(f"Sharpness score: {reference_score:.2f}")