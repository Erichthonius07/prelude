package com.prelude.capture.capture

import com.prelude.capture.model.YuvFrame

data class FrameFeatures(val sharpness: Float, val texture: Float)

object SharpnessEstimator {

    private const val TARGET = 256 // downscale edge for speed + noise robustness

    private fun yPlaneDownsampled(frame: YuvFrame): Pair<FloatArray, Int> {
        val srcW = frame.width; val srcH = frame.height
        val w = TARGET; val h = TARGET
        val out = FloatArray(w * h)
        val xStep = srcW.toFloat() / w; val yStep = srcH.toFloat() / h
        val y = frame.y
        for (j in 0 until h) {
            val sy = (j * yStep).toInt().coerceAtMost(srcH - 1)
            for (i in 0 until w) {
                val sx = (i * xStep).toInt().coerceAtMost(srcW - 1)
                out[j * w + i] = y[sy * srcW + sx].toFloat()
            }
        }
        return out to w
    }

    /** Variance of the 3x3 Laplacian — the sharpness score. */
    fun varianceOfLaplacian(frame: YuvFrame): Float {
        val (g, w) = yPlaneDownsampled(frame); val h = w
        if (w < 3) return 0f
        var sum = 0.0; var sumSq = 0.0; var n = 0
        for (j in 1 until h - 1) for (i in 1 until w - 1) {
            val c = g[j * w + i]
            val lap = 4 * c - g[(j - 1) * w + i] - g[(j + 1) * w + i] -
                    g[j * w + i - 1] - g[j * w + i + 1]
            sum += lap; sumSq += lap * lap; n++
        }
        if (n == 0) return 0f
        val mean = sum / n
        return ((sumSq / n) - mean * mean).toFloat().coerceAtLeast(0f)
    }

    /** Sobel gradient magnitude mean — the texture/bucket key. */
    fun textureEnergy(frame: YuvFrame): Float {
        val (g, w) = yPlaneDownsampled(frame); val h = w
        if (w < 3) return 0f
        var sum = 0.0; var n = 0
        for (j in 1 until h - 1) for (i in 1 until w - 1) {
            val gx = -g[(j-1)*w+i-1] + g[(j-1)*w+i+1] - 2*g[j*w+i-1] + 2*g[j*w+i+1] - g[(j+1)*w+i-1] + g[(j+1)*w+i+1]
            val gy = -g[(j-1)*w+i-1] - 2*g[(j-1)*w+i] - g[(j-1)*w+i+1] + g[(j+1)*w+i-1] + 2*g[(j+1)*w+i] + g[(j+1)*w+i+1]
            sum += kotlin.math.sqrt(gx * gx + gy * gy); n++
        }
        return if (n == 0) 0f else (sum / n).toFloat()
    }

    fun features(frame: YuvFrame) = FrameFeatures(varianceOfLaplacian(frame), textureEnergy(frame))
}