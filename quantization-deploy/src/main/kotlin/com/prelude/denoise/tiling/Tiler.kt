package com.prelude.denoise.tiling

import kotlin.math.min

/**
 * Tiler handles breaking a full-resolution image into smaller, fixed-size tiles
 * for inference, and reassembling them.
 *
 * To avoid boundary artifacts from convolutional padding, it:
 * 1. Discards an outer [haloSize] border from each tile's inference output.
 * 2. Blends the next [blendSize] pixels linearly with adjacent tiles to hide seams.
 * 3. Uses a stride of (tileSize - 2*haloSize - blendSize) to ensure perfectly
 *    summed weights across overlaps.
 */
class Tiler(
    val tileSize: Int = 256,
    val haloSize: Int = 17, // DnCNN receptive field radius (17 conv layers, 3x3 kernels)
    val blendSize: Int = 16 // Pixels to linearly cross-fade
) {
    init {
        require(tileSize > 2 * haloSize + blendSize) { 
            "Tile size must be large enough to contain the halos and blend region." 
        }
    }

    private val stride = tileSize - 2 * haloSize - blendSize

    fun process(
        input: FloatArray,
        width: Int,
        height: Int,
        channels: Int = 3,
        inferencer: (FloatArray) -> FloatArray
    ): FloatArray {
        val output = FloatArray(width * height * channels)
        val weightSum = FloatArray(width * height)

        // Calculate number of tiles needed (ceiling division)
        val tilesX = (width + stride - 1) / stride
        val tilesY = (height + stride - 1) / stride

        val tileBuffer = FloatArray(tileSize * tileSize * channels)

        for (ty in 0..tilesY) {
            for (tx in 0..tilesX) {
                // Top-left coordinate of the tile in the full image
                var startX = tx * stride
                var startY = ty * stride

                // Shift the last tile to exactly align with the right/bottom edge
                // to avoid out-of-bounds reads and ensure full coverage.
                if (startX + tileSize > width) startX = width - tileSize
                if (startY + tileSize > height) startY = height - tileSize
                
                if (startX < 0) startX = 0
                if (startY < 0) startY = 0

                // Extract tile
                extractTile(input, width, height, channels, startX, startY, tileBuffer)

                // Run inference
                val outTile = inferencer(tileBuffer)

                // Accumulate back into output with weights
                accumulateTile(
                    outTile, output, weightSum,
                    width, height, channels, startX, startY
                )
                
                // If we hit the edge exactly due to shifting, we don't need further tiles in this row/col
                if (startX == width - tileSize) break
            }
            val checkY = ty * stride
            if (checkY >= height - tileSize) break
        }

        // Normalize by weight sum to complete the blending
        for (y in 0 until height) {
            for (x in 0 until width) {
                val w = weightSum[y * width + x]
                val outIdx = (y * width + x) * channels
                if (w > 0f) {
                    for (c in 0 until channels) {
                        output[outIdx + c] /= w
                    }
                } else {
                    // Fallback if untouched (should not happen with correct tiling math)
                    for (c in 0 until channels) {
                        output[outIdx + c] = input[outIdx + c]
                    }
                }
            }
        }

        return output
    }

    private fun extractTile(
        source: FloatArray, srcW: Int, srcH: Int, channels: Int,
        startX: Int, startY: Int,
        dest: FloatArray
    ) {
        dest.fill(0f)
        for (y in 0 until tileSize) {
            val sy = startY + y
            if (sy >= srcH) break
            for (x in 0 until tileSize) {
                val sx = startX + x
                if (sx >= srcW) break
                
                val srcIdx = (sy * srcW + sx) * channels
                val dstIdx = (y * tileSize + x) * channels
                for (c in 0 until channels) {
                    dest[dstIdx + c] = source[srcIdx + c]
                }
            }
        }
    }

    private fun accumulateTile(
        tile: FloatArray, dest: FloatArray, weightSum: FloatArray,
        destW: Int, destH: Int, channels: Int,
        startX: Int, startY: Int
    ) {
        val isLeftEdge = startX == 0
        val isRightEdge = startX + tileSize >= destW
        val isTopEdge = startY == 0
        val isBottomEdge = startY + tileSize >= destH

        for (y in 0 until tileSize) {
            val dy = startY + y
            if (dy >= destH) break
            
            val wy = getWeight1D(y, isTopEdge, isBottomEdge)
            if (wy == 0f) continue

            for (x in 0 until tileSize) {
                val dx = startX + x
                if (dx >= destW) break
                
                val wx = getWeight1D(x, isLeftEdge, isRightEdge)
                val w = wx * wy
                if (w == 0f) continue

                val srcIdx = (y * tileSize + x) * channels
                val dstIdx = (dy * destW + dx) * channels
                
                for (c in 0 until channels) {
                    dest[dstIdx + c] += tile[srcIdx + c] * w
                }
                weightSum[dy * destW + dx] += w
            }
        }
    }

    private fun getWeight1D(localPos: Int, isStartEdge: Boolean, isEndEdge: Boolean): Float {
        val dStart = if (isStartEdge) Int.MAX_VALUE else localPos
        val dEnd = if (isEndEdge) Int.MAX_VALUE else tileSize - 1 - localPos
        val dist = min(dStart, dEnd)
        
        return when {
            dist < haloSize -> 0.0f
            dist < haloSize + blendSize -> (dist - haloSize + 0.5f) / blendSize
            else -> 1.0f
        }
    }
}
