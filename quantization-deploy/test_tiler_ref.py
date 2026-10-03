"""Tests for tiler_ref.py (numpy reference of the Java Tiler).

Golden properties (rule X1/D3):
  - a stack of symmetric box-blur layers run through the tiler must match the
    full-frame result within 1e-5 relative (halo 17 >= receptive field 5)
  - images smaller than one tile take the zero-pad path and reproduce the
    input exactly under an identity inferencer
  - the T9 abandon flag stops the run between tiles
A cross-language fixture replay against the real Java Tiler is added by
TilerGoldenVectorTest (see test_tiler_ref golden_replay test).
"""

import unittest
from pathlib import Path

import numpy as np

import tiler_ref
from tiler_ref import Tiler, tiled_inference


def box_blur_stack(layers: int = 5):
    """Symmetric 3x3 box blur (zero-padded), applied `layers` times, on HWC float32."""
    def apply(img: np.ndarray) -> np.ndarray:
        out = img.astype(np.float32)
        h, w = out.shape[0], out.shape[1]
        for _ in range(layers):
            padded = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="constant")
            acc = np.zeros_like(out)
            for dy in range(3):
                for dx in range(3):
                    acc += padded[dy:dy + h, dx:dx + w, :]
            out = (acc / np.float32(9.0)).astype(np.float32)
        return out
    return apply


def relative_diff(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b))) / denom


class TilerRefTest(unittest.TestCase):

    def test_box_blur_stack_tiled_matches_full_frame(self):
        """Golden test: symmetric linear filter, tiled vs full-frame within 1e-5."""
        rng = np.random.default_rng(1234)
        img = rng.random((512, 512, 3), dtype=np.float32)
        blur = box_blur_stack(layers=5)  # receptive field radius 5 <= halo 17

        full = blur(img)
        tiled = tiled_inference(Tiler(), img, blur)

        self.assertIsNotNone(tiled)
        self.assertEqual(full.shape, tiled.shape)
        rel = relative_diff(full, tiled)
        self.assertLessEqual(rel, 1e-5, f"tiled vs full-frame relative diff {rel}")

    def test_image_smaller_than_tile_zero_pad_identity(self):
        img = np.linspace(0.0, 0.9, 100 * 80 * 3, dtype=np.float32).reshape(100, 80, 3)
        tiled = tiled_inference(Tiler(), img, lambda tile: tile)
        self.assertIsNotNone(tiled)
        self.assertEqual(tiled.shape, img.shape)
        np.testing.assert_array_equal(tiled, img)

    def test_t9_abandon_stops_between_tiles(self):
        img = np.zeros((512, 512, 3), dtype=np.float32)
        calls = {"n": 0}

        def inferencer(tile):
            calls["n"] += 1
            return tile

        # Flag already set: no tile runs at all.
        self.assertIsNone(tiled_inference(Tiler(), img, inferencer, abandon=lambda: True))
        self.assertEqual(0, calls["n"])

        # Flag set after the first tile: exactly one tile runs, result dropped.
        def abandon_after_first():
            return calls["n"] >= 1
        self.assertIsNone(tiled_inference(Tiler(), img, inferencer, abandon=abandon_after_first))
        self.assertEqual(1, calls["n"])

    def test_default_geometry_matches_java(self):
        tiler = Tiler()
        self.assertEqual(256, tiler.tileSize)
        self.assertEqual(17, tiler.haloSize)
        self.assertEqual(16, tiler.blendSize)
        self.assertEqual(206, tiler.stride)

    def test_invalid_tiler_rejected(self):
        with self.assertRaises(ValueError):
            Tiler(tileSize=8, haloSize=4, blendSize=4)

    # ---- cross-language golden replay against the real Java Tiler ----

    FIXTURE = Path(__file__).parent / "test" / "golden_tiler_vector.txt"

    @staticmethod
    def java_make_image(width: int, height: int, channels: int) -> np.ndarray:
        """Mirror of TilerGoldenVectorTest.makeImage: (x + 2y + 0.5c) / 32, flat float[]."""
        out = np.zeros(width * height * channels, dtype=np.float32)
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    out[(y * width + x) * channels + c] = np.float32(
                        (x + 2 * y + 0.5 * c) / 32.0)
        return out

    @staticmethod
    def java_box_blur_tile(tile_hwc: np.ndarray) -> np.ndarray:
        """Mirror of TilerGoldenVectorTest.boxBlurTile: TWO passes of 3x3 box
        blur (receptive-field radius 2), zero-padded, same accumulation order
        (dy, dx), float32."""
        ts, ch = tile_hwc.shape[0], tile_hwc.shape[2]
        flat = tile_hwc.reshape(-1).astype(np.float32)

        def one_pass(src: np.ndarray) -> np.ndarray:
            out = np.zeros_like(src)
            for y in range(ts):
                for x in range(ts):
                    for c in range(ch):
                        s = np.float32(0.0)
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < ts and 0 <= nx < ts:
                                    s += src[(ny * ts + nx) * ch + c]
                        out[(y * ts + x) * ch + c] = np.float32(s / np.float32(9.0))
            return out

        tmp = one_pass(flat)
        out = one_pass(tmp)
        return out.reshape(ts, ts, ch)

    def test_golden_replay_against_java_fixture(self):
        """tiler_ref output must match the committed Java Tiler output (<= 1e-5)."""
        self.assertTrue(self.FIXTURE.exists(), f"missing fixture {self.FIXTURE}")
        lines = self.FIXTURE.read_text().splitlines()
        cases = 0
        line_i = 0
        while line_i < len(lines):
            line = lines[line_i]
            if line.startswith("case "):
                parts = line.split()
                name = parts[1]
                params = dict(p.split("=") for p in parts[2:])
                ts, halo, blend = (int(params["tileSize"]), int(params["haloSize"]),
                                   int(params["blendSize"]))
                w, h, c = int(params["width"]), int(params["height"]), int(params["channels"])
                flat_in = np.array(lines[line_i + 1].split()[1:], dtype=np.float32)
                expected = np.array(lines[line_i + 2].split()[1:], dtype=np.float32)
                self.assertEqual(w * h * c, flat_in.size)
                input_hwc = self.java_make_image(w, h, c).reshape(h, w, c)
                self.assertTrue(np.array_equal(flat_in, input_hwc.reshape(-1)),
                                f"{name}: input construction must match Java bit-for-bit")

                got = tiled_inference(Tiler(ts, halo, blend), input_hwc, self.java_box_blur_tile)
                self.assertIsNotNone(got)
                max_diff = float(np.max(np.abs(got.reshape(-1) - expected)))
                self.assertLessEqual(max_diff, 1e-5,
                                     f"{name}: tiler_ref vs Java max abs diff {max_diff}")
                cases += 1
                line_i += 3
            else:
                line_i += 1
        self.assertEqual(2, cases, "fixture must contain both cases")


if __name__ == "__main__":
    unittest.main()
