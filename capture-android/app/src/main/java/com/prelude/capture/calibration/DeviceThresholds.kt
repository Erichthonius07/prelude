package com.prelude.capture.calibration

data class DeviceThresholds(
    val lowTextureBlurFloor: Double = 0.011,
    val mediumTextureBlurFloor: Double = 0.018,
    val highTextureBlurFloor: Double = 0.026,
    val alignmentConfidenceFloor: Double = 0.72,
    val minimumUsableFrames: Int = 6,
    val inferenceTimeoutMs: Long = 500L,
    val calibratedAt: String = "2026-10-18",
    // Added add-only: texture-energy bucket edges (mean gradient magnitude, luminance 0..1) from the calibration tool.
    val lowTextureEdge: Double = 0.03,
    val highTextureEdge: Double = 0.08,
)

interface ThresholdCalculator {
    fun blurThreshold(textureBucket: TextureBucket): Double
    fun timeoutBudgetMs(p95LatencyMs: Long): Long
}

enum class TextureBucket { LOW, MEDIUM, HIGH }
