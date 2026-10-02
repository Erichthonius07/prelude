import cv2
import csv
import numpy as np
import os

DATASET_DIR = r"C:\Prelude\dataset"
INPUT_CSV = r"C:\Prelude\all_alignment_results.csv"
OUTPUT_CSV = r"C:\Prelude\classifier_features.csv"

def gradient_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(sobel_x, sobel_y)

    gradient_mean = float(np.mean(magnitude))
    gradient_std = float(np.std(magnitude))

    magnitude = np.clip(magnitude, 0, 1024)

    hist, _ = np.histogram(
        magnitude,
        bins=16,
        range=(0, 1024)
    )

    hist = hist.astype(np.float64)
    hist = hist / hist.sum()

    return gradient_mean, gradient_std, hist


with open(INPUT_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

output_rows = []

for i, row in enumerate(rows, start=1):

    burst_id = row["burst_id"]
    frame = row["frame"]

    image_path = os.path.join(
        DATASET_DIR,
        burst_id,
        frame
    )

    image = cv2.imread(image_path)

    if image is None:
        print(f"ERROR: Could not read {image_path}")
        continue

    gradient_mean, gradient_std, hist = gradient_features(image)

    output_row = {
        "burst_id": burst_id,
        "frame": frame,
        "reference_frame": row["reference_frame"],
        "laplacian_variance": row["frame_sharpness"],
        "orb_keypoint_count": row["keypoint_count"],
        "gradient_mean": gradient_mean,
        "gradient_std": gradient_std,
        "ransac_inlier_ratio": row["ransac_inlier_ratio"]
    }

    for j in range(16):
        output_row[f"gradient_hist_{j+1:02d}"] = hist[j]

    output_rows.append(output_row)

    print(f"[{i}/{len(rows)}] {burst_id} - {frame}")


fieldnames = [
    "burst_id",
    "frame",
    "reference_frame",
    "laplacian_variance",
    "orb_keypoint_count",
    "gradient_mean",
    "gradient_std"
]

fieldnames += [
    f"gradient_hist_{i:02d}"
    for i in range(1, 17)
]

fieldnames.append("ransac_inlier_ratio")


with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(output_rows)


print("\nFEATURE EXTRACTION COMPLETE")
print(f"Total samples: {len(output_rows)}")
print(f"Features saved to: {OUTPUT_CSV}")