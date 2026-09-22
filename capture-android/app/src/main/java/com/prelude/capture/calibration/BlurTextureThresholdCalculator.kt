package com.prelude.capture.calibration

import com.prelude.capture.capture.FrameFeatures
import kotlin.math.max

/**
 * Bucketed blur threshold via F1 maximization.
 *
 * Deterministic procedure (focus-sweep calibration):
 *  1. Capture N frames while sweeping focus; compute (texture, sharpness) per frame.
 *  2. Bucket frames by texture quantiles (low-texture scenes have lower achievable sharpness).
 *  3. Within each bucket, pseudo-label: a focus sweep is unimodal in sharpness, so frames near the
 *     bucket's sharpness peak are "sharp" (positive), the rest "blurry" (negative).
 *  4. Choose the per-bucket sharpness threshold that maximizes F1 separating positive/negative.
 */
class BlurTextureThresholdCalculator(
    private val repo: CalibrationRepository,
) : ThresholdCalculator {

    override val type = ThresholdType.BLUR_TEXTURE
    override val blockedReason: String? = null

    companion object {
        // FLAGGED DEFAULTS — confirm with orchestrator.
        const val NUM_BUCKETS = 4
        const val POSITIVE_FRACTION = 0.80f // sharpness >= fraction*bucketMax -> "sharp"
        const val MIN_SAMPLES = 16
    }

    override suspend fun calibrate(samples: List<FrameFeatures>): CalibrationOutcome {
        if (samples.size < MIN_SAMPLES) {
            return CalibrationOutcome.Blocked("Need >= $MIN_SAMPLES sweep samples, got ${samples.size}.")
        }
        val threshold = compute(samples, NUM_BUCKETS)
        repo.saveBlurTexture(threshold)
        return CalibrationOutcome.Success(
            "Buckets=${threshold.bucketEdges.size + 1}, " +
            "sharpness floors=${threshold.bucketSharpnessMin.map { "%.1f".format(it) }}"
        )
    }

    internal fun compute(samples: List<FrameFeatures>, numBuckets: Int): BlurTextureThreshold {
        val byTexture = samples.sortedBy { it.texture }
        val bucketSize = max(1, byTexture.size / numBuckets)
        val edges = (1 until numBuckets).map { byTexture[(it * bucketSize).coerceAtMost(byTexture.size - 1)].texture }

        val thresholds = (0 until numBuckets).map { b ->
            val lo = if (b == 0) Float.NEGATIVE_INFINITY else edges[b - 1]
            val hi = if (b == numBuckets - 1) Float.POSITIVE_INFINITY else edges[b]
            val frames = byTexture.filter { it.texture > lo && it.texture <= hi }
            if (frames.isEmpty()) 0f
            else {
                val maxSharp = frames.maxOf { it.sharpness }
                val labeled = frames.map { it.sharpness to (it.sharpness >= maxSharp * POSITIVE_FRACTION) }
                bestF1Threshold(labeled)
            }
        }
        return BlurTextureThreshold(edges, thresholds)
    }

    private fun bestF1Threshold(labeled: List<Pair<Float, Boolean>>): Float {
        val candidates = labeled.map { it.first }.distinct().sorted()
        var best = candidates.firstOrNull() ?: 0f
        var bestF1 = -1f
        for (c in candidates) {
            val tp = labeled.count { it.first >= c && it.second }
            val fp = labeled.count { it.first >= c && !it.second }
            val fn = labeled.count { it.first < c && it.second }
            val precision = if (tp + fp > 0) tp.toFloat() / (tp + fp) else 0f
            val recall = if (tp + fn > 0) tp.toFloat() / (tp + fn) else 0f
            val f1 = if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0f
            if (f1 > bestF1) { bestF1 = f1; best = c }
        }
        return best
    }
}