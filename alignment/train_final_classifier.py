import csv
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

INPUT_CSV = r"C:\Prelude\classifier_dataset.csv"
OUTPUT_MODEL = r"C:\Prelude\alignment_classifier.pkl"

FEATURES = [
    "laplacian_variance",
    "orb_keypoint_count",
    "gradient_mean",
    "gradient_std"
] + [f"gradient_hist_{i:02d}" for i in range(1, 17)]

X = []
y = []

with open(INPUT_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        X.append([
            float(row[feature])
            for feature in FEATURES
        ])
        y.append(int(row["alignment_label"]))

X = np.array(X)
y = np.array(y)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

model.fit(X, y)

with open(OUTPUT_MODEL, "wb") as f:
    pickle.dump(model, f)

print("===== FINAL CLASSIFIER TRAINED =====")
print(f"Samples  : {len(X)}")
print(f"Features : {X.shape[1]}")
print(f"Good     : {np.sum(y == 1)}")
print(f"Bad      : {np.sum(y == 0)}")
print(f"Model    : {OUTPUT_MODEL}")

print("\n===== SAMPLE PREDICTIONS =====")

probabilities = model.predict_proba(X)[:, 1]

for i in range(min(10, len(X))):
    prediction = 1 if probabilities[i] >= 0.5 else 0

    print(
        f"Sample {i+1:02d} | "
        f"Actual={y[i]} | "
        f"Predicted={prediction} | "
        f"Good Probability={probabilities[i]:.4f}"
    )