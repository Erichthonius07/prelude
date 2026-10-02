import sys
from pathlib import Path

import torch
import numpy as np

# Add the training directory to the path so we can import model.py
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root / "training"))
from model import DnCNN

import litert_torch
import ai_edge_litert.interpreter as tflite

def compute_psnr(pt_tensor: np.ndarray, tfl_tensor: np.ndarray) -> float:
    mse = np.mean((pt_tensor - tfl_tensor) ** 2)
    if mse == 0:
        return float('inf')
    # Using 1.0 as the data_range assuming normalized or feature map bounds
    # Since these are unbounded feature outputs (raw DnCNN logits), PSNR is just 10 * log10( max_val^2 / mse )
    # Let's just use 1.0 for typical normalized image range, or max over the tensor.
    # For a robust parity check, max absolute difference is better. We'll report PSNR assuming range of 1.0.
    return float(-10.0 * np.log10(mse))

def main():
    print("=== Prelude 5a: Conversion Smoke Test & Parity Gate ===")
    
    # 1. Instantiate untrained DnCNN
    model = DnCNN(in_channels=3, out_channels=3, num_features=64, num_layers=17)
    model.eval()
    print("[1/4] Instantiated untrained DnCNN (dncnn-untrained-smoke).")

    # 2. Generate dummy input (batch=1, channels=3, h=256, w=256)
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    print(f"[2/4] Generated dummy input tensor of shape {dummy_input.shape}.")

    # 3. Convert to LiteRT
    print("[3/4] Tracing and converting to LiteRT FP32...")
    edge_model = litert_torch.convert(model, (dummy_input,))
    
    tflite_path = "dncnn-untrained-smoke.tflite"
    edge_model.export(tflite_path)
    print(f"      Saved LiteRT model to {tflite_path}")

    # 4. Parity Gate
    print("[4/4] Running Parity Gate...")
    
    # PyTorch inference
    with torch.no_grad():
        pt_output = model(dummy_input).numpy()

    # LiteRT inference
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], dummy_input.numpy())
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])

    # Compare
    max_abs_diff = np.max(np.abs(pt_output - tflite_output))
    psnr = compute_psnr(pt_output, tflite_output)
    
    print(f"\nParity Results:")
    print(f"  PyTorch Output Shape: {pt_output.shape}")
    print(f"  LiteRT Output Shape:  {tflite_output.shape}")
    print(f"  Max Abs Diff:         {max_abs_diff:.6e}")
    print(f"  PSNR (data_range=1):  {psnr:.2f} dB")

    # Tolerance thresholds
    TOLERANCE = 1e-4
    PSNR_MIN = 100.0

    if max_abs_diff > TOLERANCE or psnr < PSNR_MIN:
        print("\n[FAIL] Parity Gate Failed: Numerical differences exceed tolerance.")
        sys.exit(1)
    else:
        print("\n[PASS] Parity Gate Passed: PyTorch and LiteRT outputs match.")
        sys.exit(0)

if __name__ == "__main__":
    main()
