# training/export_tflite.py
from pathlib import Path
import tensorflow as tf

def export_to_tflite(
    checkpoint_path: str = "dncnn_best_med.keras",
    output_tflite_path: str = "dncnn_baseline.tflite"
):
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found. Run train.py first.")

    model = tf.keras.models.load_model(ckpt_file)

    # Convert Keras model directly to LiteRT (TFLite) FP32 with dynamic spatial dimensions
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Export complete. LiteRT FP32 model saved to '{output_tflite_path}' ({len(tflite_model)} bytes).")

if __name__ == "__main__":
    export_to_tflite()