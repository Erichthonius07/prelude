"""Tests for evaluate.py (offline FP32/INT8 quality evaluation).

Covers the task's requirements:
  - analytic PSNR: identity fp32 model vs ground truth = input + 0.1 -> exactly
    20 dB with data_range=1.0 (MSE = 0.01)
  - fp32/int8 rows are distinguishable and correctly labeled; FP32-vs-INT8
    agreement recorded separately
  - unreadable image / missing pair / size mismatch / ambiguous encoding ->
    refused BEFORE any model is loaded, no output written
  - provenance fields required (incl. computed SHA-256 and tool versions)
  - output feeds submit_metrics.py dry-run successfully (HTTP mocked)

Model files are never needed: load_interpreter is patched with fakes.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import evaluate as ev
import submit_metrics
from evaluate import EvalError, main
from evaluate import evaluate as run_evaluate


def gradient_image(h: int = 48, w: int = 64) -> np.ndarray:
    """Deterministic HWC gradient in [0, 0.8] so gt = input + 0.1 stays in [0,1]."""
    flat = (np.arange(h * w * 3, dtype=np.float32) % 251) / np.float32(251.0)
    return (flat * np.float32(0.8)).reshape(h, w, 3)


def make_npy_pair(pairs_dir: Path, stem: str, gt_offset: float = 0.1,
                  h: int = 48, w: int = 64):
    inp = gradient_image(h, w)
    gt = inp + np.float32(gt_offset)
    np.save(pairs_dir / "input" / f"{stem}.npy", inp)
    np.save(pairs_dir / "ground_truth" / f"{stem}.npy", gt)
    return inp, gt


def make_pairs_dir(tmp: Path, stems: tuple[str, ...] = ("img0",)) -> Path:
    pairs_dir = tmp / "pairs"
    (pairs_dir / "input").mkdir(parents=True)
    (pairs_dir / "ground_truth").mkdir(parents=True)
    for stem in stems:
        make_npy_pair(pairs_dir, stem)
    return pairs_dir


class FakeInterpreter:
    """Stands in for an ai_edge_litert Interpreter (tile-level transform)."""

    def __init__(self, transform, in_dtype=np.float32, out_dtype=np.float32,
                 scale=(1.0, 0)):
        shape = np.array([1, 3, 256, 256])
        self._in = [{"index": 0, "dtype": in_dtype, "shape": shape,
                     "quantization": scale}]
        self._out = [{"index": 0, "dtype": out_dtype, "shape": shape,
                      "quantization": scale}]
        self.transform = transform

    def get_input_details(self):
        return self._in

    def get_output_details(self):
        return self._out

    def set_tensor(self, index, value):
        self._value = value

    def invoke(self):
        x = self._value
        if np.dtype(self._in[0]["dtype"]) != np.float32:
            s, z = self._in[0]["quantization"][:2]
            x = (x.astype(np.float32) - np.float32(z)) * np.float32(s)
        y = self.transform(x)
        if np.dtype(self._out[0]["dtype"]) != np.float32:
            s, z = self._out[0]["quantization"][:2]
            info = np.iinfo(self._out[0]["dtype"])
            y = np.clip(np.round(y / np.float32(s)) + z, info.min, info.max)
            y = y.astype(self._out[0]["dtype"])
        self._result = y

    def get_tensor(self, index):
        return self._result


def run_evaluate_with_fakes(pairs_dir: Path, tmp: Path, fp32_offset: float = 0.0,
                            int8_offset: float = 0.05, **kwargs) -> dict:
    fp32_path = Path(tmp) / "fp32.tflite"
    int8_path = Path(tmp) / "int8.tflite"
    fp32_path.write_bytes(b"dummy")
    int8_path.write_bytes(b"dummy")
    fp32_fake = FakeInterpreter(lambda x: x + np.float32(fp32_offset))
    int8_fake = FakeInterpreter(lambda x: x + np.float32(int8_offset))
    with mock.patch.object(ev, "load_interpreter", side_effect=[fp32_fake, int8_fake]):
        return run_evaluate(
            fp32_path, int8_path, pairs_dir,
            domain=kwargs.get("domain", "linear"), input_kind="sidd-noisy",
            checkpoint_sha256="a" * 64, dataset="unit-test-fixture",
            smoke=kwargs.get("smoke", False))


def row(metrics: dict, image_id: str, precision: str) -> dict:
    matches = [r for r in metrics["per_image"]
               if r["imageId"] == image_id and r["precision"] == precision]
    assert len(matches) == 1, f"expected exactly one {precision} row for {image_id}"
    return matches[0]


class EvaluateTest(unittest.TestCase):

    def test_analytic_psnr_identity_fp32_is_exactly_20dB(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs_dir = make_pairs_dir(Path(tmp))
            metrics = run_evaluate_with_fakes(pairs_dir, Path(tmp))

            fp32_row = row(metrics, "img0", "fp32")
            self.assertAlmostEqual(20.0, fp32_row["psnr_gt"], places=3,
                                   msg="identity model vs gt=input+0.1: MSE=0.01 -> PSNR=20 dB")
            self.assertGreater(fp32_row["ssim_gt"], 0.0)
            self.assertLessEqual(fp32_row["ssim_gt"], 1.0)

    def test_int8_row_distinguishable_and_agreement_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs_dir = make_pairs_dir(Path(tmp))
            metrics = run_evaluate_with_fakes(pairs_dir, Path(tmp))

            fp32_row = row(metrics, "img0", "fp32")
            int8_row = row(metrics, "img0", "int8")
            # int8 fake = input + 0.05 -> MSE 0.0025 -> ~26.02 dB vs gt=input+0.1
            self.assertAlmostEqual(26.0206, int8_row["psnr_gt"], places=2)
            self.assertLess(fp32_row["psnr_gt"], int8_row["psnr_gt"],
                            "fp32 (0.1 off) must score worse than int8 (0.05 off)")

            agreement = metrics["agreement_fp32_vs_int8"][0]
            self.assertEqual("img0", agreement["imageId"])
            # fp32 output vs int8 output differ by exactly 0.05 -> ~26.02 dB
            self.assertAlmostEqual(26.0206, agreement["psnr"], places=2)
            self.assertNotIn("psnr_gt", agreement,
                             "agreement rows must be separate from vs-ground-truth rows")

    def test_errors_refuse_before_any_model_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def exploding_loader(path):
                raise AssertionError(f"model must not be loaded for {path}")

            # size mismatch
            bad = tmp / "mismatch"
            (bad / "input").mkdir(parents=True)
            (bad / "ground_truth").mkdir()
            np.save(bad / "input" / "img0.npy", gradient_image(48, 64))
            np.save(bad / "ground_truth" / "img0.npy", gradient_image(64, 48))
            with mock.patch.object(ev, "load_interpreter", exploding_loader):
                with self.assertRaisesRegex(EvalError, "size mismatch"):
                    run_evaluate(Path("f.tflite"), Path("i.tflite"), bad, "linear",
                                 "sidd-noisy", "a" * 64, "d")

            # missing counterpart
            missing = tmp / "missing"
            (missing / "input").mkdir(parents=True)
            (missing / "ground_truth").mkdir()
            np.save(missing / "input" / "img0.npy", gradient_image())
            with mock.patch.object(ev, "load_interpreter", exploding_loader):
                with self.assertRaisesRegex(EvalError, "pair mismatch"):
                    run_evaluate(Path("f.tflite"), Path("i.tflite"), missing, "linear",
                                 "sidd-noisy", "a" * 64, "d")

            # unreadable image
            garbage = tmp / "garbage"
            (garbage / "input").mkdir(parents=True)
            (garbage / "ground_truth").mkdir()
            (garbage / "input" / "img0.png").write_bytes(b"this is not an image")
            np.save(garbage / "ground_truth" / "img0.npy", gradient_image())
            with mock.patch.object(ev, "load_interpreter", exploding_loader):
                with self.assertRaisesRegex(EvalError, "unreadable"):
                    run_evaluate(Path("f.tflite"), Path("i.tflite"), garbage, "linear",
                                 "sidd-noisy", "a" * 64, "d")

            # 8-bit raster with linear domain -> ambiguous, refused
            ambig = tmp / "ambiguous"
            (ambig / "input").mkdir(parents=True)
            (ambig / "ground_truth").mkdir()
            Image.new("RGB", (8, 8), (128, 64, 255)).save(ambig / "input" / "img0.png")
            np.save(ambig / "ground_truth" / "img0.npy", gradient_image(8, 8))
            with mock.patch.object(ev, "load_interpreter", exploding_loader):
                with self.assertRaisesRegex(EvalError, "ambiguous"):
                    run_evaluate(Path("f.tflite"), Path("i.tflite"), ambig, "linear",
                                 "sidd-noisy", "a" * 64, "d")

    def test_srgb_decoding_and_16bit_linear_decode(self):
        # sRGB 8-bit: known pixel values -> /255
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "img.png"
            Image.new("RGB", (4, 4), (128, 64, 255)).save(p)
            arr = ev.decode_image(p, "srgb")
            self.assertAlmostEqual(128 / 255.0, float(arr[0, 0, 0]), places=6)
            self.assertAlmostEqual(64 / 255.0, float(arr[0, 0, 1]), places=6)
            self.assertAlmostEqual(1.0, float(arr[0, 0, 2]), places=6)
            # 16-bit grayscale PNG decodes /65535 (2D; pair validation then
            # refuses a non-3-channel input, so this path is decode-only)
            p16 = Path(tmp) / "img16.png"
            Image.new("I;16", (4, 4), 32768).save(p16)
            arr16 = ev.decode_image(p16, "linear")
            self.assertAlmostEqual(32768 / 65535.0, float(arr16[0, 0]), places=6)

    def test_int8_quantized_io_roundtrip(self):
        """int8 details: inputs are quantized before invoke, outputs dequantized."""
        scale, zp = (1.0 / 127.0, 0)
        fake = FakeInterpreter(lambda x: x, in_dtype=np.int8, out_dtype=np.int8,
                               scale=(scale, zp))
        with tempfile.TemporaryDirectory() as tmp:
            pairs_dir = make_pairs_dir(Path(tmp))
            f32p = pairs_dir.parent / "q_fp32.tflite"
            i8p = pairs_dir.parent / "q_int8.tflite"
            f32p.write_bytes(b"dummy")
            i8p.write_bytes(b"dummy")
            with mock.patch.object(ev, "load_interpreter", return_value=fake):
                metrics = run_evaluate(f32p, i8p, pairs_dir,
                                       "linear", "sidd-noisy", "a" * 64, "d")
            # quantize->identity->dequantize round trip: quantization noise
            # (step 1/127) is ~5e-6 in MSE terms, so PSNR stays within 0.1 dB
            # of the exact 20 dB — a broken round trip would collapse far away.
            int8_row = row(metrics, "img0", "int8")
            self.assertAlmostEqual(20.0, int8_row["psnr_gt"], delta=0.1)

    def test_provenance_fields_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs_dir = make_pairs_dir(Path(tmp), stems=("img0", "img1"))
            metrics = run_evaluate_with_fakes(pairs_dir, Path(tmp), smoke=True)

            self.assertEqual("cnn-v1", metrics["model_variant"])
            self.assertEqual("unit-test-fixture", metrics["dataset"])
            self.assertEqual("linear", metrics["input_domain"])
            self.assertEqual("sidd-noisy", metrics["input_kind"])
            self.assertTrue(metrics["smoke"])
            self.assertEqual("a" * 64, metrics["checkpoint_sha256"])
            self.assertEqual(["img0", "img1"], metrics["image_ids"])
            self.assertIsInstance(metrics["file_sizes"]["int8_tflite_bytes"], int)
            for algo in ("fp32_tflite", "int8_tflite"):
                digest = metrics["sha256"][algo]
                self.assertEqual(64, len(digest))
                self.assertTrue(all(c in "0123456789abcdef" for c in digest))
            for tool in ("python", "numpy", "scikit_image", "pillow", "ai_edge_litert"):
                self.assertIn(tool, metrics["tool_versions"])
                self.assertNotEqual("NOT INSTALLED", metrics["tool_versions"][tool])
            self.assertEqual(4, len(metrics["per_image"]))  # 2 images x 2 precisions
            self.assertEqual(2, len(metrics["agreement_fp32_vs_int8"]))

    def test_output_feeds_submit_metrics_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pairs_dir = make_pairs_dir(tmp)
            metrics = run_evaluate_with_fakes(pairs_dir, tmp)
            metrics_path = tmp / "metrics.json"
            metrics_path.write_text(json.dumps(metrics))

            latency_path = tmp / "latency.json"
            latency_path.write_text(json.dumps({
                "events": [
                    {"imageId": "img0", "precision": "fp32", "latencyMs": 1.5,
                     "discardRaceEvent": False, "timeoutEvent": False},
                    {"imageId": "img0", "precision": "int8", "latencyMs": 1.2,
                     "discardRaceEvent": False, "timeoutEvent": False},
                ]
            }))

            class FakeResp:
                def __init__(self, status_code, payload):
                    self.status_code = status_code
                    self._payload = payload
                def json(self):
                    return self._payload

            def fake_get(url, timeout=5):
                if url.endswith("/manifest"):
                    return FakeResp(200, {"imageIds": ["img0"]})
                return FakeResp(404, {})

            with mock.patch.object(submit_metrics.requests, "get", side_effect=fake_get):
                rc = submit_metrics.run_cli([
                    "--run-id", "smoke-unit", "--planned-test-n", "1",
                    "--planned-comparisons", "1", "--base-url", "http://mock",
                    "--metrics-json", str(metrics_path),
                    "--latency-json", str(latency_path),
                ])
            self.assertEqual(0, rc, "submit_metrics dry run must succeed on evaluate.py output")

    def test_main_writes_output_file_and_respects_smoke_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pairs_dir = make_pairs_dir(tmp)
            (tmp / "f.tflite").write_bytes(b"dummy")
            (tmp / "i.tflite").write_bytes(b"dummy")
            out = tmp / "out.json"
            fp32_fake = FakeInterpreter(lambda x: x)
            int8_fake = FakeInterpreter(lambda x: x + np.float32(0.05))
            with mock.patch.object(ev, "load_interpreter", side_effect=[fp32_fake, int8_fake]):
                rc = main([
                "--fp32", str(tmp / "f.tflite"), "--int8", str(tmp / "i.tflite"),
                "--pairs-dir", str(pairs_dir), "--input-domain", "linear",
                "--input-kind", "sidd-noisy", "--checkpoint-sha256", "a" * 64,
                "--dataset", "unit-test-fixture", "--out", str(out), "--smoke",
            ])
            self.assertEqual(0, rc)
            data = json.loads(out.read_text())
            self.assertTrue(data["smoke"])

            # main() with no models present must fail with exit 2 and no output
            out2 = tmp / "out2.json"
            rc2 = main([
                "--fp32", "missing.tflite", "--int8", "missing.tflite",
                "--pairs-dir", str(pairs_dir), "--input-domain", "linear",
                "--input-kind", "sidd-noisy", "--checkpoint-sha256", "a" * 64,
                "--dataset", "unit-test-fixture", "--out", str(out2),
            ])
            self.assertEqual(2, rc2)
            self.assertFalse(out2.exists())


if __name__ == "__main__":
    unittest.main()
