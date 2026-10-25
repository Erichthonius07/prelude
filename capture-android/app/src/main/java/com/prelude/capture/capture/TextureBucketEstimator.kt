package com.prelude.capture.capture

import android.graphics.Bitmap
import android.graphics.Color
import com.prelude.capture.calibration.DeviceThresholds
import com.prelude.capture.calibration.TextureBucket
import kotlin.math.sqrt

/**
 * Buckets a frame by local texture energy (mean gradient magnitude on a downsampled luminance grid), so its
 * sharpness is judged against the matching bucket's blur threshold instead of one global cutoff (spec 4.2).
 * The bucket edges come from the device's calibration profile.
 */
class TextureBucketEstimator(private val thresholds: DeviceThresholds) {
    fun meanGradient(bitmap: Bitmap): Double {
        if (bitmap.width < 3 || bitmap.height < 3) return 0.0
        val stepX = (bitmap.width / 64).coerceAtLeast(1)
        val stepY = (bitmap.height / 64).coerceAtLeast(1)
        var sum = 0.0
        var count = 0
        for (y in stepY until bitmap.height - stepY step stepY) {
            for (x in stepX until bitmap.width - stepX step stepX) {
                val gx = luma(bitmap.getPixel(x + stepX, y)) - luma(bitmap.getPixel(x - stepX, y))
                val gy = luma(bitmap.getPixel(x, y + stepY)) - luma(bitmap.getPixel(x, y - stepY))
                sum += sqrt(gx * gx + gy * gy)
                count++
            }
        }
        return if (count == 0) 0.0 else sum / count
    }

    fun bucket(bitmap: Bitmap): TextureBucket {
        val g = meanGradient(bitmap)
        return when {
            g < thresholds.lowTextureEdge -> TextureBucket.LOW
            g > thresholds.highTextureEdge -> TextureBucket.HIGH
            else -> TextureBucket.MEDIUM
        }
    }

    private fun luma(c: Int): Double = (0.2126 * Color.red(c) + 0.7152 * Color.green(c) + 0.0722 * Color.blue(c)) / 255.0
}
