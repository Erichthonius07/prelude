import csv
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

INPUT_CSV = r"C:\Prelude\classifier_dataset.csv"

with open(INPUT_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

feature_names = [
    "laplacian_variance",
    "orb_keypoint_count",
    "gradient_mean",
    "gradient_std"
]

feature_names += [
    f"gradient_hist_{i:02d}"
    for i in range(1, 17)
]

X = np.array([
    [float(row[name]) for name in feature_names]
    for row in rows
])

y = np.array([
    int(row["alignment_label"])
    for row in rows
])

groups = np.array([
    row["burst_id"]
    for row in rows
])

print("===== DATASET =====")
print(f"Samples  : {len(X)}")
print(f"Features : {X.shape[1]}")
print(f"Good     : {np.sum(y == 1)}")
print(f"Bad      : {np.sum(y == 0)}")
print(f"Bursts   : {len(np.unique(groups))}")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

train_groups = groups[train_idx]
test_groups = groups[test_idx]

print("\n===== SPLIT =====")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

print(
    "Training bursts  :",
    ", ".join(sorted(np.unique(train_groups)))
)

print(
    "Testing bursts   :",
    ", ".join(sorted(np.unique(test_groups)))
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)
recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n===== RESULTS =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Bad", "Good"],
        zero_division=0
    )
)

print("\n===== SAMPLE PREDICTIONS =====")

for idx, prediction in zip(test_idx, y_pred):
    print(
        f"{rows[idx]['burst_id']} | "
        f"{rows[idx]['frame']} | "
        f"Actual={y[idx]} | "
        f"Predicted={prediction}"
    )