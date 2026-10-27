# training/inference_med.py
import os
import platform

# 1. Silence TensorFlow C++ oneDNN and CPU instruction logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 2. On native Windows, enable Keras 3 GPU acceleration via CUDA if available
if platform.system() == "Windows":
    try:
        import torch
        if torch.cuda.is_available():
            os.environ["KERAS_BACKEND"] = "torch"
    except ImportError:
        pass

import math
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

def to_numpy(tensor) -> np.ndarray:
    """Safely converts any backend tensor (PyTorch CUDA, TF, or Keras) to a CPU NumPy array."""
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return keras.ops.convert_to_numpy(tensor)

def estimate_blind_noise_sigma(img_arr: np.ndarray) -> float:
    """Immerkaer's Fast Noise Variance Estimation using a 3x3 Laplacian operator[cite: 70]."""
    # Ensure input is a 4D TensorFlow tensor [1, H, W, 3] on CPU for metric calculation
    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)
    
    with tf.device("/CPU:0"):
        img_tf = tf.convert_to_tensor(img_arr, dtype=tf.float32)
        gray = tf.image.rgb_to_grayscale(img_tf) * 255.0

        kernel = tf.constant([
            [ 1.0, -2.0,  1.0],
            [-2.0,  4.0, -2.0],
            [ 1.0, -2.0,  1.0]
        ], dtype=tf.float32)
        kernel = tf.reshape(kernel, [3, 3, 1, 1])

        conv = tf.nn.conv2d(gray, kernel, strides=[1, 1, 1, 1], padding="VALID")
        h, w = int(conv.shape[1]), int(conv.shape[2])
        sigma = tf.reduce_sum(tf.abs(conv)) * math.sqrt(0.5 * math.pi) / (6.0 * float(w * h))
        return float(sigma.numpy())

def run_inference(
    image_path: str = "noisy_image.jpg",
    model_path: str = "dncnn_best_med.keras",
    output_report: str = "denoise_analysis_report_med.jpg",
    max_dim: int = 512
):
    model = keras.models.load_model(model_path)

    orig_img = Image.open(image_path).convert("RGB")
    w, h = orig_img.size

    # Prevent RTX 3050 VRAM OOM by scaling down high-resolution frames for inference testing
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        orig_img = orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"Resized test image from {w}x{h} to {new_w}x{new_h} to fit within GPU VRAM.")

    orig_arr = np.asarray(orig_img, dtype=np.float32) / 255.0
    input_batch = np.expand_dims(orig_arr, axis=0)

    # Run model prediction and safely move output from GPU to CPU NumPy array
    raw_output = model(input_batch, training=False)
    pred_np = to_numpy(raw_output)

    # Clip values to valid image range [0.0, 1.0]
    denoised_arr = np.clip(pred_np[0], 0.0, 1.0)
    noise_map = orig_arr - denoised_arr

    orig_sigma = estimate_blind_noise_sigma(orig_arr)
    den_sigma = estimate_blind_noise_sigma(denoised_arr)
    reduction_pct = ((orig_sigma - den_sigma) / orig_sigma) * 100.0 if orig_sigma > 0 else 0.0

    vis_noise = np.clip(noise_map * 3.0 + 0.5, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(orig_arr)
    axes[0].set_title(f"Original (Noise σ: {orig_sigma:.2f})")
    axes[0].axis("off")

    axes[1].imshow(denoised_arr)
    axes[1].set_title(f"Denoised (Noise σ: {den_sigma:.2f})")
    axes[1].axis("off")

    axes[2].imshow(vis_noise)
    axes[2].set_title(f"Extracted AI Noise Map\n(Reduction: {reduction_pct:.1f}%)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_report, dpi=200)
    print(f"Saved analysis report to {output_report}")

if __name__ == "__main__":
    run_inference()