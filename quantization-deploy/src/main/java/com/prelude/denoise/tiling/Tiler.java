package com.prelude.denoise.tiling;

/**
 * Tiler handles breaking a full-resolution image into smaller, fixed-size tiles
 * for inference, and reassembling them.
 *
 * To avoid boundary artifacts from convolutional padding, it:
 *   1. Discards an outer haloSize border from each tile's inference output.
 *   2. Blends the next blendSize pixels linearly with adjacent tiles to hide seams.
 *   3. Uses a stride of (tileSize - 2*haloSize - blendSize) to ensure perfectly
 *      summed weights across overlaps.
 *
 * float[] is like a flat numpy float32 array. Java doesn't have native
 * multi-dimensional array slicing, so we do index math manually (y * width + x).
 */
public final class Tiler {

    private final int tileSize;
    private final int haloSize;   // DnCNN receptive field radius (17 conv layers, 3x3 kernels)
    private final int blendSize;  // Pixels to linearly cross-fade
    private final int stride;

    /** Functional interface for the inference callback (like Python's Callable). */
    public interface Inferencer {
        float[] run(float[] tile);
    }

    public Tiler(int tileSize, int haloSize, int blendSize) {
        if (tileSize <= 2 * haloSize + blendSize) {
            throw new IllegalArgumentException(
                "Tile size must be large enough to contain the halos and blend region.");
        }
        this.tileSize = tileSize;
        this.haloSize = haloSize;
        this.blendSize = blendSize;
        this.stride = tileSize - 2 * haloSize - blendSize;
    }

    /** Default: tileSize=256, haloSize=17, blendSize=16 */
    public Tiler() {
        this(256, 17, 16);
    }

    public int getTileSize()  { return tileSize; }
    public int getHaloSize()  { return haloSize; }
    public int getBlendSize() { return blendSize; }
    public int getStride()    { return stride; }

    /**
     * Process a full image through tiled inference with halo-drop and linear blending.
     *
     * @param input      flat pixel array [H * W * C]
     * @param width      image width
     * @param height     image height
     * @param channels   number of channels (default 3)
     * @param inferencer callback that runs inference on a single tile
     * @return blended output array, same shape as input
     */
    public float[] process(float[] input, int width, int height, int channels,
                           Inferencer inferencer) {
        float[] output = new float[width * height * channels];
        float[] weightSum = new float[width * height];
        float[] tileBuffer = new float[tileSize * tileSize * channels];

        int tilesX = (width + stride - 1) / stride;
        int tilesY = (height + stride - 1) / stride;

        for (int ty = 0; ty <= tilesY; ty++) {
            for (int tx = 0; tx <= tilesX; tx++) {
                // Top-left coordinate of the tile in the full image
                int startX = tx * stride;
                int startY = ty * stride;

                // Shift the last tile to exactly align with the right/bottom edge
                if (startX + tileSize > width)  startX = width - tileSize;
                if (startY + tileSize > height) startY = height - tileSize;

                if (startX < 0) startX = 0;
                if (startY < 0) startY = 0;

                // Extract tile from full image (zero-padded for edge cases)
                extractTile(input, width, height, channels, startX, startY, tileBuffer);

                // Run inference on the tile
                float[] outTile = inferencer.run(tileBuffer);

                // Accumulate into output with blending weights
                accumulateTile(outTile, output, weightSum,
                               width, height, channels, startX, startY);

                if (startX == width - tileSize) break;
            }
            int checkY = ty * stride;
            if (checkY >= height - tileSize) break;
        }

        // Normalize by weight sum to complete the blending
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float w = weightSum[y * width + x];
                int outIdx = (y * width + x) * channels;
                if (w > 0f) {
                    for (int c = 0; c < channels; c++) {
                        output[outIdx + c] /= w;
                    }
                } else {
                    // Fallback if untouched (should not happen with correct tiling math)
                    for (int c = 0; c < channels; c++) {
                        output[outIdx + c] = input[outIdx + c];
                    }
                }
            }
        }
        return output;
    }

    // Note: images smaller than one tile are zero-padded to the tile size.
    // Edge pixels may differ from a same-size full-frame run due to receptive field bleed.
    // Real frames are larger than a tile.
    private void extractTile(float[] source, int srcW, int srcH, int channels,
                             int startX, int startY, float[] dest) {
        java.util.Arrays.fill(dest, 0f);
        for (int y = 0; y < tileSize; y++) {
            int sy = startY + y;
            if (sy >= srcH) break;
            for (int x = 0; x < tileSize; x++) {
                int sx = startX + x;
                if (sx >= srcW) break;
                int srcIdx = (sy * srcW + sx) * channels;
                int dstIdx = (y * tileSize + x) * channels;
                for (int c = 0; c < channels; c++) {
                    dest[dstIdx + c] = source[srcIdx + c];
                }
            }
        }
    }

    private void accumulateTile(float[] tile, float[] dest, float[] weightSum,
                                int destW, int destH, int channels,
                                int startX, int startY) {
        boolean isLeftEdge   = startX == 0;
        boolean isRightEdge  = startX + tileSize >= destW;
        boolean isTopEdge    = startY == 0;
        boolean isBottomEdge = startY + tileSize >= destH;

        for (int y = 0; y < tileSize; y++) {
            int dy = startY + y;
            if (dy >= destH) break;

            float wy = getWeight1D(y, isTopEdge, isBottomEdge);
            if (wy == 0f) continue;

            for (int x = 0; x < tileSize; x++) {
                int dx = startX + x;
                if (dx >= destW) break;

                float wx = getWeight1D(x, isLeftEdge, isRightEdge);
                float w = wx * wy;
                if (w == 0f) continue;

                int srcIdx = (y * tileSize + x) * channels;
                int dstIdx = (dy * destW + dx) * channels;

                for (int c = 0; c < channels; c++) {
                    dest[dstIdx + c] += tile[srcIdx + c] * w;
                }
                weightSum[dy * destW + dx] += w;
            }
        }
    }

    private float getWeight1D(int localPos, boolean isStartEdge, boolean isEndEdge) {
        int dStart = isStartEdge ? Integer.MAX_VALUE : localPos;
        int dEnd   = isEndEdge   ? Integer.MAX_VALUE : tileSize - 1 - localPos;
        int dist   = Math.min(dStart, dEnd);

        if (dist < haloSize) return 0.0f;
        if (dist < haloSize + blendSize) return (dist - haloSize + 0.5f) / blendSize;
        return 1.0f;
    }
}
