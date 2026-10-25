package com.prelude.pipeline.fusion

import android.graphics.Bitmap
import android.os.SystemClock
import com.prelude.pipeline.contract.AlignedFrameSet
import com.prelude.pipeline.contract.FusedFrame
import com.prelude.pipeline.contract.FusionStrategy
import com.prelude.pipeline.contract.FusionStrategyMetrics
import com.prelude.pipeline.contract.PipelineMode
import com.prelude.pipeline.modules.FusionWeightModel

/**
 * Role 3: four-way fusion over an [AlignedFrameSet].
 *
 * All four strategies run on every multi-frame burst so they can be compared on the same input. Live bursts
 * have no ground truth, so only latency is reported here; PSNR/SSIM per strategy come from the batch runner on
 * the held-out test set. Confidence weighting reads [com.prelude.pipeline.contract.AlignedFrame.ransacInlierRatio]
 * only, per the data contract.
 */
class FusionModule(private val learnedModel: FusionWeightModel? = null) {

    class Result(
        val outputs: Map<FusionStrategy, FusedFrame>,
        val metrics: List<FusionStrategyMetrics>,
        /** The frame handed on to DenoiseModule. */
        val chosen: FusedFrame,
    )

    fun fuse(set: AlignedFrameSet, mode: PipelineMode, preferred: FusionStrategy): Result {
        if (mode == PipelineMode.FUSION_SINGLE) {
            val ref = set.reference
            val passThrough = FusedFrame(set.burstId, ref.bitmap, mode, strategy = null, framesUsed = 1)
            val skipped = FusionStrategy.values().map { FusionStrategyMetrics(it, available = false, latencyMs = 0, framesUsed = 1, note = "skipped in fusion_single") }
            return Result(emptyMap(), skipped, passThrough)
        }

        val ref = set.reference.bitmap
        val w = ref.width; val h = ref.height
        val frames = set.usableFrames.filter { it.bitmap.width == w && it.bitmap.height == h }
        val pixels = frames.map { f -> IntArray(w * h).also { f.bitmap.getPixels(it, 0, w, 0, 0, w, h) } }
        val ransac = DoubleArray(frames.size) { frames[it].ransacInlierRatio }

        val outputs = LinkedHashMap<FusionStrategy, FusedFrame>()
        val metrics = ArrayList<FusionStrategyMetrics>()
        for (strategy in FusionStrategy.values()) {
            val start = SystemClock.elapsedRealtime()
            val fused: IntArray? = when (strategy) {
                FusionStrategy.NAIVE_MEAN -> weightedMean(pixels, DoubleArray(pixels.size) { 1.0 })
                FusionStrategy.TRIMMED_MEAN -> trimmedMean(pixels)
                FusionStrategy.CONFIDENCE_WEIGHTED -> weightedMean(pixels, ransac)
                FusionStrategy.LEARNED -> learnedModel?.let { learned(pixels, it.weights(pixels, w, h, ransac)) }
            }
            val ms = SystemClock.elapsedRealtime() - start
            if (fused == null) {
                metrics += FusionStrategyMetrics(strategy, available = false, latencyMs = 0, framesUsed = frames.size, note = "learned fusion model not connected")
            } else {
                outputs[strategy] = FusedFrame(set.burstId, Bitmap.createBitmap(fused, w, h, Bitmap.Config.ARGB_8888), mode, strategy, frames.size)
                metrics += FusionStrategyMetrics(strategy, available = true, latencyMs = ms, framesUsed = frames.size, note = learnedModel?.modelId?.takeIf { strategy == FusionStrategy.LEARNED })
            }
        }
        val chosen = outputs[preferred] ?: outputs[FusionStrategy.CONFIDENCE_WEIGHTED] ?: outputs.values.first()
        return Result(outputs, metrics, chosen)
    }

    private fun weightedMean(frames: List<IntArray>, weights: DoubleArray): IntArray {
        val total = weights.sum()
        val norm = DoubleArray(weights.size) { if (total > 0) weights[it] / total else 1.0 / weights.size }
        val out = IntArray(frames[0].size)
        for (i in out.indices) {
            var r = 0.0; var g = 0.0; var b = 0.0
            for (k in frames.indices) {
                val c = frames[k][i]; val wk = norm[k]
                r += wk * ((c shr 16) and 0xFF); g += wk * ((c shr 8) and 0xFF); b += wk * (c and 0xFF)
            }
            out[i] = pack(r, g, b)
        }
        return out
    }

    /** Per-pixel, per-channel mean with the min and max sample dropped (needs ≥ 3 frames, else plain mean). */
    private fun trimmedMean(frames: List<IntArray>): IntArray {
        val n = frames.size
        if (n < 3) return weightedMean(frames, DoubleArray(n) { 1.0 })
        val out = IntArray(frames[0].size)
        for (i in out.indices) {
            var rs = 0; var gs = 0; var bs = 0
            var rMin = 255; var gMin = 255; var bMin = 255
            var rMax = 0; var gMax = 0; var bMax = 0
            for (k in 0 until n) {
                val c = frames[k][i]
                val r = (c shr 16) and 0xFF; val g = (c shr 8) and 0xFF; val b = c and 0xFF
                rs += r; gs += g; bs += b
                if (r < rMin) rMin = r; if (r > rMax) rMax = r
                if (g < gMin) gMin = g; if (g > gMax) gMax = g
                if (b < bMin) bMin = b; if (b > bMax) bMax = b
            }
            val d = (n - 2).toDouble()
            out[i] = pack((rs - rMin - rMax) / d, (gs - gMin - gMax) / d, (bs - bMin - bMax) / d)
        }
        return out
    }

    private fun learned(frames: List<IntArray>, planes: List<FloatArray>): IntArray {
        require(planes.size == frames.size) { "learned fusion model returned ${planes.size} weight planes for ${frames.size} frames" }
        val out = IntArray(frames[0].size)
        for (i in out.indices) {
            var sum = 0.0
            for (k in frames.indices) sum += planes[k][i].coerceAtLeast(0f)
            var r = 0.0; var g = 0.0; var b = 0.0
            for (k in frames.indices) {
                val wk = if (sum > 0) planes[k][i].coerceAtLeast(0f) / sum else 1.0 / frames.size
                val c = frames[k][i]
                r += wk * ((c shr 16) and 0xFF); g += wk * ((c shr 8) and 0xFF); b += wk * (c and 0xFF)
            }
            out[i] = pack(r, g, b)
        }
        return out
    }

    private fun pack(r: Double, g: Double, b: Double): Int =
        (0xFF shl 24) or (r.toInt().coerceIn(0, 255) shl 16) or (g.toInt().coerceIn(0, 255) shl 8) or b.toInt().coerceIn(0, 255)
}
