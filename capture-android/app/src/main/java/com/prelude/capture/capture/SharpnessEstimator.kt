package com.prelude.capture.capture

import android.graphics.Bitmap
import kotlin.math.pow

class SharpnessEstimator {
    /** Approximate variance-of-Laplacian score on a downsampled luminance grid. */
    fun varianceOfLaplacian(bitmap: Bitmap): Double {
        if (bitmap.width < 3 || bitmap.height < 3) return 0.0
        val stepX = (bitmap.width / 64).coerceAtLeast(1)
        val stepY = (bitmap.height / 64).coerceAtLeast(1)
        val values = mutableListOf<Double>()
        for (y in stepY until bitmap.height - stepY step stepY) {
            for (x in stepX until bitmap.width - stepX step stepX) {
                val center = luminance(bitmap.getPixel(x, y))
                val left = luminance(bitmap.getPixel(x - stepX, y))
                val right = luminance(bitmap.getPixel(x + stepX, y))
                val up = luminance(bitmap.getPixel(x, y - stepY))
                val down = luminance(bitmap.getPixel(x, y + stepY))
                values += (left + right + up + down - 4.0 * center)
            }
        }
        val mean = values.average().takeIf { !it.isNaN() } ?: return 0.0
        return values.map { (it - mean).pow(2) }.average()
    }

    private fun luminance(color: Int): Double =
        (0.2126 * android.graphics.Color.red(color) + 0.7152 * android.graphics.Color.green(color) + 0.0722 * android.graphics.Color.blue(color)) / 255.0
}
