package com.prelude.capture.calibration

class BlurTextureThresholdCalculator(
    private val thresholds: DeviceThresholds,
) : ThresholdCalculator {
    override fun blurThreshold(textureBucket: TextureBucket): Double = when (textureBucket) {
        TextureBucket.LOW -> thresholds.lowTextureBlurFloor
        TextureBucket.MEDIUM -> thresholds.mediumTextureBlurFloor
        TextureBucket.HIGH -> thresholds.highTextureBlurFloor
    }

    override fun timeoutBudgetMs(p95LatencyMs: Long): Long = minOf(
        thresholds.inferenceTimeoutMs,
        (p95LatencyMs * 1.2).toLong(),
    )
}
