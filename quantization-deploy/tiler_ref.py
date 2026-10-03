"""tiler_ref.py — numpy reference of the Java Tiler (quantization-deploy/tiling/Tiler.java).

Mirrors the Java semantics exactly (same loop order, same float32 math) so the
offline evaluation in evaluate.py processes images the same way the on-device
tiled path does:

  - tileSize 256, haloSize 17 (DnCNN receptive-field radius), blendSize 16,
    stride = tileSize - 2*haloSize - blendSize (rule D3)
  - last tile column/row is shifted to align with the right/bottom edge
  - tiles smaller than the image are extracted with zero padding (image < tile)
  - separable 1D weights: dist < halo -> 0; dist < halo+blend ->
    (dist - halo + 0.5) / blend; else 1  (halo-drop + linear cross-fade)
  - output normalized by the summed weights; pixels with zero total weight
    fall back to the input value (same fallback as Java)
  - T9: an abandon flag is checked BEFORE each tile; True stops between tiles
    and returns None (result dropped). The tile in flight always completes.

All arithmetic is float32 to track the Java float math. Do NOT insert any
gamma/domain conversion here (rule C3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

Inferencer = Callable[[np.ndarray], np.ndarray]
AbandonFlag = Callable[[], bool]

DEFAULT_TILE_SIZE = 256
DEFAULT_HALO_SIZE = 17   # DnCNN: 17 conv layers of 3x3 -> receptive-field radius 17
DEFAULT_BLEND_SIZE = 16


@dataclass
class Tiler:
    tileSize: int = DEFAULT_TILE_SIZE
    haloSize: int = DEFAULT_HALO_SIZE
    blendSize: int = DEFAULT_BLEND_SIZE

    def __post_init__(self) -> None:
        if self.tileSize <= 2 * self.haloSize + self.blendSize:
            raise ValueError(
                "Tile size must be large enough to contain the halos and blend region.")
        self.stride = self.tileSize - 2 * self.haloSize - self.blendSize


def tiled_inference(tiler: Tiler, input_hwc: np.ndarray, inferencer: Inferencer,
                    abandon: Optional[AbandonFlag] = None) -> Optional[np.ndarray]:
    """Run a full HWC image through tiled inference; mirrors Tiler.process.

    Returns the blended float32 output, or None when the run was abandoned
    between tiles (T9).
    """
    if input_hwc.ndim != 3:
        raise ValueError(f"input must be HWC (3 dims), got shape {input_hwc.shape}")
    height, width, channels = input_hwc.shape
    inp = np.ascontiguousarray(input_hwc, dtype=np.float32)
    tile = tiler.tileSize
    stride = tiler.stride

    output = np.zeros((height, width, channels), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    tiles_x = (width + stride - 1) // stride
    tiles_y = (height + stride - 1) // stride

    for ty in range(tiles_y + 1):
        for tx in range(tiles_x + 1):
            # T9: cooperative abandon point — between tiles, never mid-tile.
            if abandon is not None and abandon():
                return None

            # Top-left coordinate of the tile in the full image
            start_x = tx * stride
            start_y = ty * stride

            # Shift the last tile to exactly align with the right/bottom edge
            if start_x + tile > width:
                start_x = width - tile
            if start_y + tile > height:
                start_y = height - tile
            if start_x < 0:
                start_x = 0
            if start_y < 0:
                start_y = 0

            tile_in = _extract_tile(inp, width, height, start_x, start_y, tile)
            tile_out = np.asarray(inferencer(tile_in), dtype=np.float32)
            if tile_out.shape != tile_in.shape:
                raise ValueError(
                    f"inferencer returned shape {tile_out.shape}, expected {tile_in.shape}")

            _accumulate_tile(tile_out, output, weight_sum, tiler, start_x, start_y)

            if start_x == width - tile:
                break
        if ty * stride >= height - tile:
            break

    # Normalize by the summed weights; zero-weight pixels fall back to the input.
    nonzero = weight_sum > 0
    safe_w = np.where(nonzero, weight_sum, np.float32(1.0))
    blended = output / safe_w[..., None]
    output = np.where(nonzero[..., None], blended, inp).astype(np.float32)
    return output


def _extract_tile(src: np.ndarray, src_w: int, src_h: int,
                  start_x: int, start_y: int, tile: int) -> np.ndarray:
    """Copy a tile out of the image; out-of-image areas stay zero (Java extractTile)."""
    channels = src.shape[2]
    dest = np.zeros((tile, tile, channels), dtype=np.float32)
    copy_w = min(tile, src_w - start_x)
    copy_h = min(tile, src_h - start_y)
    dest[:copy_h, :copy_w, :] = src[start_y:start_y + copy_h, start_x:start_x + copy_w, :]
    return dest


def _accumulate_tile(tile_out: np.ndarray, output: np.ndarray, weight_sum: np.ndarray,
                     tiler: Tiler, start_x: int, start_y: int) -> None:
    """Add the tile's weighted contribution into the output (Java accumulateTile)."""
    tile = tiler.tileSize
    dest_h, dest_w = output.shape[0], output.shape[1]
    is_left_edge = start_x == 0
    is_right_edge = start_x + tile >= dest_w
    is_top_edge = start_y == 0
    is_bottom_edge = start_y + tile >= dest_h

    copy_h = min(tile, dest_h - start_y)
    copy_w = min(tile, dest_w - start_x)

    wx = _weights_1d(tile, tiler.haloSize, tiler.blendSize, copy_w,
                     is_left_edge, is_right_edge)
    wy = _weights_1d(tile, tiler.haloSize, tiler.blendSize, copy_h,
                     is_top_edge, is_bottom_edge)
    w = wy[:, None] * wx[None, :]

    region = output[start_y:start_y + copy_h, start_x:start_x + copy_w, :]
    region += tile_out[:copy_h, :copy_w, :] * w[..., None]
    weight_sum[start_y:start_y + copy_h, start_x:start_x + copy_w] += w


def _weights_1d(tile: int, halo: int, blend: int, count: int,
                is_start_edge: bool, is_end_edge: bool) -> np.ndarray:
    """Vectorized Java getWeight1D for the first `count` positions of a tile axis."""
    pos = np.arange(tile, dtype=np.float32)
    # Java sentinel: Integer.MAX_VALUE when the tile touches that image edge.
    far = np.float32(2 ** 31)
    dist_start = np.full(tile, far, dtype=np.float32) if is_start_edge else pos
    dist_end = np.full(tile, far, dtype=np.float32) if is_end_edge else np.float32(tile - 1) - pos
    dist = np.minimum(dist_start, dist_end)

    zero = dist < halo
    mid = (dist < halo + blend) & ~zero
    w = np.where(mid, (dist - np.float32(halo) + np.float32(0.5)) / np.float32(blend),
                 np.where(zero, np.float32(0.0), np.float32(1.0))).astype(np.float32)
    return w[:count]
