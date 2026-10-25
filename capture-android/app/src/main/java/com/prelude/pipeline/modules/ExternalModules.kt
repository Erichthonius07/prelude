package com.prelude.pipeline.modules

import android.os.SystemClock
import com.prelude.capture.calibration.DeviceThresholds
import com.prelude.capture.model.FrameBurst
import com.prelude.pipeline.contract.AlignedFrame
import com.prelude.pipeline.contract.AlignedFrameSet
import com.prelude.pipeline.contract.AlignmentMetrics
import com.prelude.pipeline.contract.DenoisedFrame
import com.prelude.pipeline.contract.FusedFrame

// ---------------------------------------------------------------------------------------------------------------
// CONNECTION POINTS for modules owned by other roles. Each one has a "not connected" implementation that takes the
// pipeline's own defined fallback path, so the app runs end to end today and nothing shown is made up.
// Swap the implementation in PipelineFactory once the real module lands.
// ---------------------------------------------------------------------------------------------------------------

/**
 * Role 2: ORB/FAST + RANSAC homography alignment.
 *
 * Contract: align only [FrameBurst.acceptedFrames] against [referenceIndex] (the sharpest frame). Every output frame
 * carries RANSAC's real inlier ratio and is `accepted` only when that ratio ≥ [DeviceThresholds.alignmentConfidenceFloor].
 * The learned classifier's predictions go in [AlignmentMetrics.classifierPredictions], never into the frames.
 */
interface AlignmentModule {
    fun align(burst: FrameBurst, referenceIndex: Int, thresholds: DeviceThresholds): AlignedFrameSet
}

/**
 * Placeholder until Role 2's module is connected. It doesn't invent confidence values: only the reference frame
 * is passed through (identity homography), every other frame is left unaligned with a NaN inlier ratio
 * ("not attempted"), so the decider takes the single-frame fallback path.
 */
class NotConnectedAlignment : AlignmentModule {
    override fun align(burst: FrameBurst, referenceIndex: Int, thresholds: DeviceThresholds): AlignedFrameSet {
        val start = SystemClock.elapsedRealtime()
        val frames = burst.acceptedFrames.mapNotNull { f ->
            val bitmap = f.bitmap ?: return@mapNotNull null
            val isRef = f.index == referenceIndex
            AlignedFrame(f.index, bitmap, IDENTITY.copyOf(), if (isRef) 1.0 else Double.NaN, accepted = isRef)
        }
        return AlignedFrameSet(
            burstId = burst.burstId,
            referenceIndex = referenceIndex,
            frames = frames,
            metrics = AlignmentMetrics(referenceIndex, attempted = 0, succeeded = 0, inlierRatios = emptyList(), latencyMs = SystemClock.elapsedRealtime() - start, note = "alignment module not connected"),
        )
    }

    companion object { val IDENTITY = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f) }
}

/**
 * Role 5a (+ Role 4's trained model): TFLite INT8 denoiser with NNAPI delegate and CPU fallback.
 *
 * Contract: must return within [timeoutMs] by running inference on its own thread and racing it against a timer
 * (discard-race, spec 4.3). If the timer wins, return the input frame with `timedOut = true` and drop the late result
 * when it eventually arrives. The orchestrator never cancels or overrides this.
 */
interface DenoiseModule {
    fun denoise(fused: FusedFrame, timeoutMs: Long): DenoisedFrame
}

/** Placeholder until the deployed denoiser is connected: returns the fused frame unchanged and says so. */
class NotConnectedDenoise : DenoiseModule {
    override fun denoise(fused: FusedFrame, timeoutMs: Long): DenoisedFrame =
        DenoisedFrame(fused.burstId, fused.bitmap, applied = false, timedOut = false, latencyMs = 0, backend = "none", modelId = null)
}

/**
 * Role 3 (yours): the learned per-pixel fusion-weighting model, the 4th fusion strategy.
 *
 * Contract: inputs are the aligned frames' pixels plus RANSAC inlier ratios (the parameter is typed and named so the
 * classifier's prediction can't be passed by accident). Returns one non-negative weight plane per frame, each of
 * length width × height; FusionModule normalises them per pixel.
 */
interface FusionWeightModel {
    val modelId: String
    fun weights(frames: List<IntArray>, width: Int, height: Int, ransacInlierRatios: DoubleArray): List<FloatArray>
}
