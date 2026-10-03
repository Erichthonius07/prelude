#!/usr/bin/env python3
"""evaluate.py — offline FP32-vs-INT8 quality evaluation (rules Q1/Q4/C3/C4).

Loads two LiteRT models (FP32 + INT8) on CPU, runs FULL images through a numpy
reference of the on-device tiler (tiler_ref.py: tile 256, halo 17, blend 16),
and computes per-image PSNR/SSIM against ground truth for BOTH precisions plus
a clearly separated FP32-vs-INT8 agreement block. Writes the metrics JSON that
submit_metrics.py consumes.

Hard rules implemented here:
  - NO gamma/sRGB/scaling conversion inside this script (rule C3). Decoding is
    fixed by --input-domain:
      srgb   : 8-bit PNG/JPG -> float32 / 255 (SIDD convention; loud note)
      linear : .npy float32 passthrough (no scaling) or true 16-bit PNG / 65535
               (loud note). 8-bit images with --input-domain linear are REFUSED
               (ambiguous encoding). NOTE: Pillow cannot faithfully decode
               16-bit RGB PNGs, so .npy is the only fully supported linear
               container for 3-channel pairs today.
  - Metrics = skimage PSNR/SSIM, data_range=1.0, channel_axis=2 (rule Q4,
    matches fusion-postprocess-demo/common/metrics.py), outputs clamped to
    [0, 1], full-resolution.
  - Refuses to run (before ANY model is loaded) if an image is unreadable, a
    pair is missing, or input/GT sizes mismatch.
  - No hard-coded metric values anywhere; provenance is recorded, never
    invented. Latency is NOT measured or claimed here (device-only, rule Q4).

Example:
  python evaluate.py --fp32 m_fp32.tflite --int8 m_int8.tflite \
      --pairs-dir data/pairs --input-domain linear --input-kind fused-frame \
      --checkpoint-sha256 <64 hex> --dataset sidd-val --out metrics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import tiler_ref
from tiler_ref import Tiler, tiled_inference

MODEL_VARIANT_DEFAULT = "cnn-v1"  # contract constant (A3/C6) — overridable via --model-variant
RASTER_EXTS = {".png", ".jpg", ".jpeg"}
ALL_EXTS = RASTER_EXTS | {".npy"}

METRIC_DATA_RANGE = 1.0  # rule Q4; mutation checks temporarily flip this to prove the tests guard it


class EvalError(Exception):
    """Fatal evaluation problem — printed to stderr, exit code 2, no output file."""


# ---------------------------------------------------------------------------
# Image decoding (fixed by --input-domain; never converted afterwards, C3)
# ---------------------------------------------------------------------------

def decode_image(path: Path, domain: str) -> np.ndarray:
    """Decode one image file to float32 HWC (or HW for single-channel 16-bit PNG).

    No value transformation beyond the documented decode; no clamping here.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim not in (2, 3):
            raise EvalError(f"{path.name}: .npy must be HWC or HW, got shape {arr.shape}")
        return arr.astype(np.float32)

    try:
        img = Image.open(path)
        img.load()
    except Exception as e:  # unreadable -> refuse (task requirement)
        raise EvalError(f"{path.name}: unreadable image ({e})") from e

    if suffix in RASTER_EXTS and img.mode in ("I;16", "I"):
        arr = np.asarray(img).astype(np.float32) / np.float32(65535.0)
        print(f"NOTE: {path.name}: 16-bit PNG decoded /65535 (linear convention, no other conversion).")
        return arr

    if domain == "linear":
        raise EvalError(
            f"{path.name}: 8-bit image with --input-domain linear is an ambiguous encoding "
            f"(rule C3) — use .npy float32 (or a true 16-bit PNG). Refusing to guess.")

    if img.mode == "RGB":
        arr = np.asarray(img, dtype=np.float32) / np.float32(255.0)
        print(f"NOTE: {path.name}: 8-bit image decoded /255 (SIDD sRGB convention, no other conversion).")
        return arr
    if img.mode == "L":
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / np.float32(255.0)
        print(f"NOTE: {path.name}: 8-bit grayscale expanded to RGB and decoded /255 (sRGB convention).")
        return arr

    raise EvalError(f"{path.name}: unsupported image mode {img.mode!r} for domain {domain!r}")


# ---------------------------------------------------------------------------
# Pair discovery & validation (all before any model is loaded)
# ---------------------------------------------------------------------------

def discover_pairs(pairs_dir: Path, domain: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Find, decode and validate all pairs. Refuses (before any model loads) on
    missing counterparts, unreadable files, size mismatches, or ambiguous
    domain/encoding combinations. Returns sorted (stem, input, ground_truth)."""
    input_dir = pairs_dir / "input"
    gt_dir = pairs_dir / "ground_truth"
    if not input_dir.is_dir() or not gt_dir.is_dir():
        raise EvalError(f"{pairs_dir}: expected subdirectories 'input/' and 'ground_truth/'")

    def by_stem(d: Path) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in ALL_EXTS:
                stem = p.stem
                if stem in found:
                    raise EvalError(f"{d}: duplicate stem '{stem}' ({found[stem].name} vs {p.name})")
                found[stem] = p
        return found

    inputs = by_stem(input_dir)
    gts = by_stem(gt_dir)
    if not inputs:
        raise EvalError(f"{input_dir}: no supported images ({', '.join(sorted(ALL_EXTS))})")

    missing_gt = sorted(set(inputs) - set(gts))
    missing_input = sorted(set(gts) - set(inputs))
    if missing_gt or missing_input:
        raise EvalError(
            f"pair mismatch in {pairs_dir}: inputs without ground truth: {missing_gt}; "
            f"ground truths without input: {missing_input}")

    pairs = []
    for stem in sorted(inputs):
        try:
            inp = decode_image(inputs[stem], domain)
        except EvalError:
            raise
        except Exception as e:
            raise EvalError(f"{inputs[stem].name}: unreadable ({e})") from e
        try:
            gt = decode_image(gts[stem], domain)
        except EvalError:
            raise
        except Exception as e:
            raise EvalError(f"{gts[stem].name}: unreadable ({e})") from e
        if inp.shape != gt.shape:
            raise EvalError(
                f"{stem}: size mismatch — input {inputs[stem].name} {inp.shape} vs "
                f"ground truth {gts[stem].name} {gt.shape}")
        if inp.ndim != 3 or inp.shape[2] != 3:
            raise EvalError(f"{stem}: input must be HWC with 3 channels, got {inp.shape}")
        pairs.append((stem, inp.astype(np.float32), gt.astype(np.float32)))
    return pairs


# ---------------------------------------------------------------------------
# LiteRT model runner (fakeable in tests; CPU only)
# ---------------------------------------------------------------------------

def load_interpreter(path: Path):
    import ai_edge_litert.interpreter as tflite  # same import as convert_smoke.py
    interpreter = tflite.Interpreter(model_path=str(path), num_threads=4)
    interpreter.allocate_tensors()
    return interpreter


def _quantize(x: np.ndarray, details: dict) -> np.ndarray:
    dtype = details["dtype"]
    if np.dtype(dtype) == np.float32:
        return np.ascontiguousarray(x, dtype=np.float32)
    scale, zero_point = details["quantization"][:2]
    info = np.iinfo(dtype)
    q = np.clip(np.round(x / np.float32(scale)) + int(zero_point), info.min, info.max)
    return np.ascontiguousarray(q.astype(dtype))


def _dequantize(y: np.ndarray, details: dict) -> np.ndarray:
    if np.dtype(details["dtype"]) == np.float32:
        return y.astype(np.float32)
    scale, zero_point = details["quantization"][:2]
    return (y.astype(np.float32) - np.float32(zero_point)) * np.float32(scale)


def run_model_full_image(interpreter, image_hwc: np.ndarray, tiler: Tiler) -> Optional[np.ndarray]:
    """Run one full HWC image through the tiled path; returns None if abandoned (T9)."""
    inp_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    shape = list(inp_details["shape"])  # e.g. [1, 3, 256, 256] (NCHW from the PyTorch conversion)
    if len(shape) != 4 or shape[0] != 1:
        raise EvalError(f"model input shape {shape} not supported (need 1xCxHxW)")
    model_c, model_h, model_w = int(shape[1]), int(shape[2]), int(shape[3])
    if image_hwc.shape[2] != model_c:
        raise EvalError(f"image has {image_hwc.shape[2]} channels, model expects {model_c}")
    if tiler.tileSize != model_h or model_h != model_w:
        raise EvalError(
            f"tiler tileSize {tiler.tileSize} != model input {model_h}x{model_w} (rule C5: fixed shapes)")

    def tile_runner(tile_hwc: np.ndarray) -> np.ndarray:
        chw = np.transpose(tile_hwc, (2, 0, 1))[None, ...]  # HWC -> 1xCxHxW (NCHW)
        interpreter.set_tensor(inp_details["index"], _quantize(chw, inp_details))
        interpreter.invoke()
        out = interpreter.get_tensor(out_details["index"])
        hwc = np.transpose(_dequantize(out, out_details)[0], (1, 2, 0))
        return hwc

    return tiled_inference(tiler, image_hwc, tile_runner)


# ---------------------------------------------------------------------------
# Metrics (rule Q4) + provenance
# ---------------------------------------------------------------------------

def compute_metrics(output: np.ndarray, ground_truth: np.ndarray) -> dict:
    out = np.clip(output, 0.0, 1.0)  # Q4: outputs clamped to [0,1]; GT used as decoded
    if ground_truth.min() < 0.0 or ground_truth.max() > 1.0:
        print(f"WARNING: ground truth outside [0,1] (min {ground_truth.min():.4f}, "
              f"max {ground_truth.max():.4f}) — metrics computed against it as decoded.")
    return {
        "psnr_gt": float(peak_signal_noise_ratio(ground_truth, out, data_range=METRIC_DATA_RANGE)),
        "ssim_gt": float(structural_similarity(ground_truth, out, data_range=METRIC_DATA_RANGE,
                                               channel_axis=2)),
    }


def agreement_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """FP32-vs-INT8 agreement — clearly separate from the vs-ground-truth rows."""
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    return {
        "psnr": float(peak_signal_noise_ratio(a, b, data_range=METRIC_DATA_RANGE)),
        "ssim": float(structural_similarity(a, b, data_range=METRIC_DATA_RANGE, channel_axis=2)),
    }


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tool_versions() -> dict:
    def ver(dist: str) -> str:
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return "NOT INSTALLED"
    return {
        "python": platform.python_version(),
        "numpy": ver("numpy"),
        "scikit_image": ver("scikit-image"),
        "pillow": ver("pillow"),
        "ai_edge_litert": ver("ai-edge-litert"),
    }


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def evaluate(fp32_path: Path, int8_path: Path, pairs_dir: Path, domain: str,
             input_kind: str, checkpoint_sha256: str, dataset: str,
             model_variant: str = MODEL_VARIANT_DEFAULT, smoke: bool = False) -> dict:
    # Validation FIRST — data problems surface before anything is loaded or run.
    pairs = discover_pairs(pairs_dir, domain)

    for p in (fp32_path, int8_path):
        if not p.is_file():
            raise EvalError(f"model file not found: {p}")
    if len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in checkpoint_sha256):
        raise EvalError("--checkpoint-sha256 must be a 64-char hex string")

    tiler = Tiler()  # tile 256, halo 17, blend 16 (rule D3, mirrors the Java defaults)
    fp32_interp = load_interpreter(fp32_path)
    int8_interp = load_interpreter(int8_path)

    per_image: list[dict] = []
    agreement: list[dict] = []
    image_ids: list[str] = []
    for stem, inp, gt in pairs:
        fp32_out = run_model_full_image(fp32_interp, inp, tiler)
        if fp32_out is None:
            raise EvalError(f"{stem}: fp32 tiled run was abandoned (T9) — cannot evaluate")
        int8_out = run_model_full_image(int8_interp, inp, tiler)
        if int8_out is None:
            raise EvalError(f"{stem}: int8 tiled run was abandoned (T9) — cannot evaluate")

        fp32_m = compute_metrics(fp32_out, gt)
        int8_m = compute_metrics(int8_out, gt)
        per_image.append({"imageId": stem, "precision": "fp32", **fp32_m})
        per_image.append({"imageId": stem, "precision": "int8", **int8_m})
        agreement.append({"imageId": stem, **agreement_metrics(fp32_out, int8_out)})
        image_ids.append(stem)
        print(f"  {stem}: fp32 psnr={fp32_m['psnr_gt']:.3f} ssim={fp32_m['ssim_gt']:.5f} | "
              f"int8 psnr={int8_m['psnr_gt']:.3f} ssim={int8_m['ssim_gt']:.5f}")

    return {
        "model_variant": model_variant,
        "file_sizes": {
            "fp32_tflite_bytes": fp32_path.stat().st_size,
            "int8_tflite_bytes": int8_path.stat().st_size,
        },
        "dataset": dataset,
        "input_domain": domain,
        "input_kind": input_kind,
        "input_size": "256x256",
        "checkpoint_sha256": checkpoint_sha256,
        "sha256": {
            "fp32_tflite": sha256_of(fp32_path),
            "int8_tflite": sha256_of(int8_path),
        },
        "smoke": bool(smoke),
        "tool_versions": tool_versions(),
        "image_ids": image_ids,
        "per_image": per_image,
        "agreement_fp32_vs_int8": agreement,
        "note": "Latency is NOT measured here (device-only, rule Q4); this file carries "
                "quality vs ground truth and FP32-vs-INT8 agreement only.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline FP32/INT8 quality evaluation (rule Q4)")
    parser.add_argument("--fp32", required=True, help="FP32 .tflite model")
    parser.add_argument("--int8", required=True, help="INT8 .tflite model")
    parser.add_argument("--pairs-dir", required=True,
                        help="directory with input/ and ground_truth/ image pairs")
    parser.add_argument("--input-domain", required=True, choices=["srgb", "linear"],
                        help="declared domain of the pairs (NO default, rule C3)")
    parser.add_argument("--input-kind", required=True, choices=["sidd-noisy", "fused-frame"],
                        help="what the input images are (provenance)")
    parser.add_argument("--checkpoint-sha256", required=True,
                        help="SHA-256 of the checkpoint the models came from (rule C4)")
    parser.add_argument("--dataset", required=True, help="dataset name (provenance)")
    parser.add_argument("--out", required=True, help="output metrics JSON path")
    parser.add_argument("--model-variant", default=MODEL_VARIANT_DEFAULT,
                        help=f"model variant (default {MODEL_VARIANT_DEFAULT}, contract A3)")
    parser.add_argument("--smoke", action="store_true",
                        help="mark the output as a smoke run (submit as smoke-* only, rule X10)")
    args = parser.parse_args(argv)

    try:
        result = evaluate(Path(args.fp32), Path(args.int8), Path(args.pairs_dir),
                          args.input_domain, args.input_kind, args.checkpoint_sha256,
                          args.dataset, args.model_variant, args.smoke)
    except EvalError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.smoke:
        print("NOTE: smoke=true — submit these numbers only under a smoke-* run id (rule X10); "
              "they are never final results.")
    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {out_path} ({len(result['per_image'])} per-image rows, "
          f"{len(result['image_ids'])} images).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
