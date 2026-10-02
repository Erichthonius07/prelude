import csv
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
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

bursts = sorted(np.unique(groups))

all_actual = []
all_predicted = []

print("===== LEAVE-ONE-BURST-OUT EVALUATION =====")

for test_burst in bursts:

    train_mask = groups != test_burst
    test_mask = groups == test_burst

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

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
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    all_actual.extend(y_test)
    all_predicted.extend(y_pred)

    print(f"\n{test_burst}")
    print(f"Samples   : {len(y_test)}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

all_actual = np.array(all_actual)
all_predicted = np.array(all_predicted)

print("\n===== OVERALL RESULTS =====")

print(
    f"Accuracy  : "
    f"{accuracy_score(all_actual, all_predicted):.4f}"
)

print(
    f"Precision : "
    f"{precision_score(all_actual, all_predicted, zero_division=0):.4f}"
)

print(
    f"Recall    : "
    f"{recall_score(all_actual, all_predicted, zero_division=0):.4f}"
)

print(
    f"F1-score  : "
    f"{f1_score(all_actual, all_predicted, zero_division=0):.4f}"
)

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(all_actual, all_predicted))

print("\n==========================================")