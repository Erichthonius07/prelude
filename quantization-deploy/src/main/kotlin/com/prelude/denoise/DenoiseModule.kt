package com.prelude.denoise

import com.prelude.denoise.model.DenoisedFrame
import com.prelude.denoise.model.FusedFrame
import com.prelude.denoise.timeout.DiscardRaceRunner
import com.prelude.denoise.timeout.InferenceRunner
import com.prelude.denoise.timeout.RaceResult
import com.prelude.denoise.timeout.SystemTimeSource
import com.prelude.denoise.timeout.TimeSource
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * Top-level denoise module: FusedFrame → DenoisedFrame (§6.3 → §6.4).
 *
 * Responsibilities:
 * - Run inference with the discard-race timeout (T1–T8)
 * - Enforce single-flight policy (A26/T6)
 * - Expose latency measurement API for calibration (D6)
 *
 * This class does NOT read or use [FusedFrame.pipelineMode] (rules D4/T5).
 *
 * @param inferenceRunner The synchronous inference call (stubbed for now).
 * @param timeSource Injectable clock (T7).
 * @param modelVariant Model name, e.g. "cnn-v1" (C6).
 * @param precision "fp32" or "int8" (Q1).
 */
class DenoiseModule(
    private val inferenceRunner: InferenceRunner,
    private val timeSource: TimeSource = SystemTimeSource,
    private val modelVariant: String = MODEL_VARIANT_CNN_V1,
    private val precision: String = PRECISION_FP32,
) {
    /** Run mode controls single-flight behavior (A26). */
    enum class RunMode {
        /** New burst while busy → fusion-only (BUSY). */
        LIVE,
        /** New burst while busy → bounded wait, then proceed. */
        BENCHMARK,
    }

    private val busyFlag = AtomicBoolean(false)
    private val lastRaceResult = AtomicReference<RaceResult>()
    private val runner = DiscardRaceRunner(inferenceRunner, timeSource)

    /**
     * Denoise a fused frame with the discard-race timeout.
     *
     * This method intentionally does NOT read [FusedFrame.pipelineMode]
     * (rules D4/T5 — `pipeline_mode` must NEVER influence the timeout).
     *
     * @param input The fused frame (§6.3).
     * @param timeoutMs Per-device calibrated timeout (T4). Must be 1..500.
     * @param mode [RunMode.LIVE] or [RunMode.BENCHMARK] (A26).
     * @return [RaceResult] containing the frame and race outcome metadata.
     */
    fun denoise(input: FusedFrame, timeoutMs: Long, mode: RunMode): RaceResult {
        require(timeoutMs in 1..DiscardRaceRunner.MAX_TIMEOUT_MS) {
            "Timeout must be 1..${DiscardRaceRunner.MAX_TIMEOUT_MS} ms (rule T4)"
        }

        // A26: benchmark mode — bounded wait for previous late inference
        if (mode == RunMode.BENCHMARK) {
            lastRaceResult.get()?.awaitLateInference(timeoutMs)
        }

        // T6: single-flight — if a previous inference is still running,
        // the interpreter must not be used concurrently (D4).
        if (!busyFlag.compareAndSet(false, true)) {
            // A26: live mode — emit fusion-only with reason BUSY
            return RaceResult.busy(
                DenoisedFrame(
                    burstId = input.burstId,
                    image = input.image.copyOf(),
                    modelVariant = modelVariant,
                    precision = precision,
                    latencyNs = 0L,
                    discardRaceOccurred = false,
                    timeoutOccurred = true,
                )
            )
        }

        return try {
            val result = runner.run(
                input = input,
                timeoutMs = timeoutMs,
                modelVariant = modelVariant,
                precision = precision,
                onLateInferenceComplete = { busyFlag.set(false) },
            )

            lastRaceResult.set(result)

            if (result.inferenceWon) {
                busyFlag.set(false)
            }
            // If timer won, busyFlag stays true until onLateInferenceComplete fires

            result
        } catch (e: Throwable) {
            busyFlag.set(false)
            throw e
        }
    }

    companion object {
        const val MODEL_VARIANT_CNN_V1 = "cnn-v1"
        const val PRECISION_FP32 = "fp32"
        const val PRECISION_INT8 = "int8"
    }
}
