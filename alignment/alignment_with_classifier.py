import os
import joblib

from alignment.feature_extraction import extract_features


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "alignment_classifier.pkl"
)


def add_classifier_confidence(aligned_result):
    model = joblib.load(MODEL_PATH)

    for frame in aligned_result["frames"]:

        if frame["frameIndex"] == aligned_result["referenceFrameIndex"]:
            frame["alignmentConfidence"] = None
            continue

        image_for_features = frame.get(
            "_originalImage",
            frame.get("image")
        )

        if image_for_features is None:
            frame["alignmentConfidence"] = 0.0
            continue

        features = extract_features(image_for_features)

        probability = model.predict_proba(
            features.reshape(1, -1)
        )[0][1]

        frame["alignmentConfidence"] = float(probability)

        if "_originalImage" in frame:
            del frame["_originalImage"]

    return aligned_result