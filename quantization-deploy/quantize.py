#!/usr/bin/env python3
"""
Role 5a — INT8 Post-Training Quantization Pipeline

Workflow steps (per /5a-quantize):
  1. Load trained PyTorch checkpoint → convert to LiteRT FP32 via litert-torch
  2. Parity gate: PyTorch vs LiteRT FP32 (max-abs-diff + PSNR)
  3. INT8 PTQ via TF Lite converter with representative data
  4. Evaluate FP32 vs INT8 on test inputs (per-image PSNR/SSIM delta)
  5. Record file sizes, report results

Input domain: sRGB [0,1] (smoke-test only — pipeline uses linear FusedFrame).
Checkpoint: training/dncnn_best_med.pth (Role 4 commit 89e3ffe)
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add training/ to path so we can import DnCNN
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))

from model import DnCNN  # noqa: E402


# ── Configuration ──────────────────────────────────────────────────
CHECKPOINT_PATH = REPO_ROOT / "training" / "dncnn_best_med.pth"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FP32_MODEL_PATH = OUTPUT_DIR / "dncnn_trained_fp32.tflite"
INT8_MODEL_PATH = OUTPUT_DIR / "dncnn_trained_int8.tflite"
RESULTS_PATH = OUTPUT_DIR / "quantization_results.json"

# Model input shape (fixed, needed for tiling — rule C5)
INPUT_SHAPE = (1, 3, 256, 256)  # NCHW
NUM_CALIBRATION_SAMPLES = 100   # for INT8 representative dataset
NUM_EVAL_SAMPLES = 20           # for FP32-vs-INT8 quality comparison
MODEL_VARIANT = "cnn-v1"        # contract §5.4/§6.4


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pytorch_model(checkpoint_path: Path) -> DnCNN:
    """Load the trained DnCNN from a .pth checkpoint."""
    model = DnCNN()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def convert_to_litert_fp32(model: DnCNN, output_path: Path) -> None:
    """Convert PyTorch model to LiteRT FP32 using litert-torch."""
    import litert_torch

    sample_input = torch.randn(*INPUT_SHAPE, dtype=torch.float32)
    edge_model = litert_torch.convert(model, (sample_input,))
    edge_model.export(str(output_path))
    print(f"  FP32 LiteRT model saved to: {output_path}")
    print(f"  FP32 file size: {output_path.stat().st_size:,} bytes")


def run_parity_gate(model: DnCNN, tflite_path: Path) -> dict:
    """
    Parity gate (rule C2): compare PyTorch vs LiteRT FP32 outputs
    on fixed inputs. STOP if they diverge.
    """
    import ai_edge_litert.interpreter as tflite

    # Fixed seed for reproducibility
    torch.manual_seed(42)
    test_input = torch.randn(*INPUT_SHAPE, dtype=torch.float32)

    # PyTorch inference
    with torch.no_grad():
        pt_output = model(test_input).numpy()

    # LiteRT inference
    interpreter = tflite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], test_input.numpy())
    interpreter.invoke()
    lt_output = interpreter.get_tensor(output_details[0]["index"])

    # Compute parity metrics
    diff = np.abs(pt_output - lt_output)
    max_abs_diff = float(np.max(diff))
    mean_abs_diff = float(np.mean(diff))

    # PSNR between the two outputs
    mse = float(np.mean((pt_output - lt_output) ** 2))
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(-10.0 * np.log10(mse))

    result = {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "psnr_db": round(psnr, 2),
        "pt_output_shape": list(pt_output.shape),
        "lt_output_shape": list(lt_output.shape),
    }

    print(f"  Max abs diff:  {max_abs_diff:.2e}")
    print(f"  Mean abs diff: {mean_abs_diff:.2e}")
    print(f"  PSNR:          {psnr:.2f} dB")

    # Parity gate: fail if max diff > 1e-4
    if max_abs_diff > 1e-4:
        print("  [FAIL] Parity gate FAILED — outputs diverge beyond tolerance.")
        print("         STOPPING per rule C2.")
        sys.exit(1)
    else:
        print("  [PASS] Parity gate passed.")

    return result


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    """
    INT8 post-training quantization using ai-edge-quantizer.
    Representative data: synthetic samples in [0,1] sRGB domain.

    NOTE (rule Q2): Final quantization MUST use real SIDD train/val
    data filtered through the leakage guard. This uses synthetic
    data for the smoke-test since SIDD images are gitignored.
    """
    from ai_edge_quantizer import Quantizer
    from ai_edge_quantizer.qtyping import TFLOperationName, QuantGranularity

    # 1. Create quantizer from the FP32 flatbuffer
    quantizer = Quantizer(str(fp32_path))

    # 2. Configure INT8 static quantization for all ops
    #    DnCNN uses: Conv2D, BatchNorm (folded into Conv), ReLU, Add (residual)
    for op_name in [
        TFLOperationName.CONV_2D,
        TFLOperationName.ADD,
        TFLOperationName.FULLY_CONNECTED,
    ]:
        try:
            quantizer.add_static_config(
                regex=".*",
                operation_name=op_name,
                activation_num_bits=8,
                weight_num_bits=8,
                weight_granularity=QuantGranularity.CHANNELWISE,
            )
        except Exception:
            pass  # Skip ops not present in the model

    # 3. Generate representative calibration data
    #    Model signature: serving_default, input name: args_0
    np.random.seed(42)
    calibration_samples = []
    for _ in range(NUM_CALIBRATION_SAMPLES):
        sample = np.random.rand(*INPUT_SHAPE).astype(np.float32)
        calibration_samples.append({"args_0": sample})

    # 4. Calibrate
    print("  Calibrating with representative data...")
    if quantizer.need_calibration:
        calibration_result = quantizer.calibrate(
            calibration_data={"serving_default": calibration_samples}
        )
    else:
        calibration_result = None

    # 5. Quantize and save
    print("  Quantizing to INT8...")
    result = quantizer.quantize(
        calibration_result=calibration_result,
        serialize_to_path=str(int8_path),
    )

    print(f"  INT8 LiteRT model saved to: {int8_path}")
    print(f"  INT8 file size: {int8_path.stat().st_size:,} bytes")


def evaluate_fp32_vs_int8(fp32_path: Path, int8_path: Path) -> list:
    """
    Evaluate FP32 vs INT8 quality on test inputs.
    Reports per-image PSNR and SSIM deltas.
    Uses skimage PSNR/SSIM, data_range=1.0, channel_axis=2 (rule Q4).
    """
    import ai_edge_litert.interpreter as tflite
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    # Load both models
    fp32_interp = tflite.Interpreter(model_path=str(fp32_path))
    fp32_interp.allocate_tensors()
    fp32_in = fp32_interp.get_input_details()
    fp32_out = fp32_interp.get_output_details()

    int8_interp = tflite.Interpreter(model_path=str(int8_path))
    int8_interp.allocate_tensors()
    int8_in = int8_interp.get_input_details()
    int8_out = int8_interp.get_output_details()

    results = []
    np.random.seed(123)  # Different seed from calibration

    for i in range(NUM_EVAL_SAMPLES):
        # Generate a test image (sRGB [0,1])
        test_input = np.random.rand(*INPUT_SHAPE).astype(np.float32)

        # FP32 inference
        fp32_interp.set_tensor(fp32_in[0]["index"], test_input)
        fp32_interp.invoke()
        fp32_output = fp32_interp.get_tensor(fp32_out[0]["index"])

        # INT8 inference (float I/O, internal INT8)
        int8_interp.set_tensor(int8_in[0]["index"], test_input)
        int8_interp.invoke()
        int8_output = int8_interp.get_tensor(int8_out[0]["index"])

        # Clamp to [0,1]
        fp32_clamped = np.clip(fp32_output, 0.0, 1.0)
        int8_clamped = np.clip(int8_output, 0.0, 1.0)

        # Convert from NCHW to NHWC for skimage
        fp32_hwc = np.transpose(fp32_clamped[0], (1, 2, 0))
        int8_hwc = np.transpose(int8_clamped[0], (1, 2, 0))

        # PSNR: FP32 output as reference, data_range=1.0 (rule Q4)
        psnr = peak_signal_noise_ratio(fp32_hwc, int8_hwc, data_range=1.0)

        # SSIM: channel_axis=2 per team metrics definition (rule Q4)
        ssim = structural_similarity(fp32_hwc, int8_hwc, data_range=1.0, channel_axis=2)

        results.append({
            "image_idx": i,
            "psnr_fp32_vs_int8": round(float(psnr), 4),
            "ssim_fp32_vs_int8": round(float(ssim), 6),
        })

        if i < 5 or psnr < 30:
            print(f"  Image {i:3d}: PSNR={psnr:.2f} dB, SSIM={ssim:.6f}")

    return results


def main():
    print("=" * 60)
    print("Role 5a — INT8 Post-Training Quantization Pipeline")
    print("=" * 60)

    # ── 0. Setup ───────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)

    ckpt_hash = sha256_file(CHECKPOINT_PATH)
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    print(f"SHA-256:    {ckpt_hash}")
    print(f"Input domain: sRGB (SMOKE-TEST ONLY)")
    print(f"Model variant: {MODEL_VARIANT}")

    # ── 1. Load PyTorch model ──────────────────────────────────
    print(f"\n[1/5] Loading trained PyTorch DnCNN...")
    model = load_pytorch_model(CHECKPOINT_PATH)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # ── 2. Convert to LiteRT FP32 ─────────────────────────────
    print(f"\n[2/5] Converting PyTorch → LiteRT FP32 (litert-torch)...")
    convert_to_litert_fp32(model, FP32_MODEL_PATH)

    # ── 3. Parity gate ────────────────────────────────────────
    print(f"\n[3/5] Running parity gate (PyTorch vs LiteRT FP32)...")
    parity = run_parity_gate(model, FP32_MODEL_PATH)

    # ── 4. INT8 quantization ──────────────────────────────────
    print(f"\n[4/5] Quantizing to INT8 PTQ...")
    print(f"  Representative samples: {NUM_CALIBRATION_SAMPLES} (synthetic sRGB [0,1])")
    print(f"  NOTE: Final run MUST use real SIDD train/val data (rule Q2)")
    quantize_int8(FP32_MODEL_PATH, INT8_MODEL_PATH)

    # ── 5. Evaluate FP32 vs INT8 ──────────────────────────────
    print(f"\n[5/5] Evaluating FP32 vs INT8 quality delta...")
    eval_results = evaluate_fp32_vs_int8(FP32_MODEL_PATH, INT8_MODEL_PATH)

    psnrs = [r["psnr_fp32_vs_int8"] for r in eval_results]
    ssims = [r["ssim_fp32_vs_int8"] for r in eval_results]
    mean_psnr = float(np.mean(psnrs))
    mean_ssim = float(np.mean(ssims))
    min_psnr = float(np.min(psnrs))
    worst_5 = sorted(eval_results, key=lambda r: r["psnr_fp32_vs_int8"])[:5]

    print(f"\n  --- FP32 vs INT8 Summary ---")
    print(f"  Mean PSNR delta: {mean_psnr:.2f} dB")
    print(f"  Mean SSIM:       {mean_ssim:.6f}")
    print(f"  Worst PSNR:      {min_psnr:.2f} dB")
    print(f"  Worst 5 images:  {[w['image_idx'] for w in worst_5]}")

    # ── 6. File sizes ─────────────────────────────────────────
    print(f"\n  --- File Sizes ---")
    fp32_size = FP32_MODEL_PATH.stat().st_size
    int8_size = INT8_MODEL_PATH.stat().st_size
    pth_size = CHECKPOINT_PATH.stat().st_size

    print(f"  PyTorch .pth:   {pth_size:>10,} bytes")
    print(f"  LiteRT FP32:    {fp32_size:>10,} bytes")
    print(f"  LiteRT INT8:    {int8_size:>10,} bytes")
    print(f"  Compression:    {fp32_size/int8_size:.1f}x")

    # ── 7. Save results ───────────────────────────────────────
    results = {
        "smoke_test": True,
        "input_domain": "srgb",
        "model_variant": MODEL_VARIANT,
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": ckpt_hash,
            "git_commit": "89e3ffece69b8ebedc50aa690ce36fa3a46d84f8",
            "run_id": "phase3-denoise-sidd-medium-l1",
        },
        "param_count": param_count,
        "parity_gate": parity,
        "file_sizes": {
            "pth_bytes": pth_size,
            "fp32_tflite_bytes": fp32_size,
            "int8_tflite_bytes": int8_size,
        },
        "int8_evaluation": {
            "num_samples": len(eval_results),
            "calibration_samples": NUM_CALIBRATION_SAMPLES,
            "calibration_source": "synthetic_srgb_smoke_test",
            "mean_psnr_fp32_vs_int8": round(mean_psnr, 4),
            "mean_ssim_fp32_vs_int8": round(mean_ssim, 6),
            "per_image": eval_results,
        },
        "latency": "NOT RUN — no Primary Device available",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_PATH}")

    print("\n" + "=" * 60)
    print(f"DONE — Smoke-test quantization complete.")
    print(f"FP32 vs INT8 quality delta: PSNR={mean_psnr:.2f} dB, SSIM={mean_ssim:.6f}")
    print(f"Latency: NOT RUN (needs Primary Device — rule D5)")
    print(f"WARNING: sRGB domain — not final numbers (rule C3)")
    print("=" * 60)


if __name__ == "__main__":
    main()
