package com.prelude.denoise.api

import com.prelude.denoise.model.FusedFrame
import com.prelude.denoise.timeout.InferenceRunner
import com.prelude.denoise.timeout.SystemTimeSource
import com.prelude.denoise.timeout.TimeSource

/**
 * Clean API for Role 1's `InferenceTimeoutCalculator` to run real inference
 * and collect per-image latencies (rule D6).
 *
 * This runs inference WITHOUT the discard-race timeout, measuring raw latency
 * for the calibration tool to compute p95 + 20% (T4).
 *
 * Usage: Role 1 calls [runInferenceForLatencyNs] repeatedly with representative
 * inputs, collects the latency distribution, then computes the calibrated
 * timeout via `DeviceThresholds.inferenceTimeoutMs`.
 *
 * No concurrent calls allowed (rule D4).
 */
class InferenceApi(
    private val inferenceRunner: InferenceRunner,
    private val timeSource: TimeSource = SystemTimeSource,
) {
    /**
     * Run a single inference and return the latency in nanoseconds.
     * This blocks until inference completes (no timeout applied).
     *
     * @param input The fused frame to denoise.
     * @return Inference latency in nanoseconds.
     */
    fun runInferenceForLatencyNs(input: FusedFrame): Long {
        val startNs = timeSource.nanoTime()
        inferenceRunner.run(input)
        return timeSource.nanoTime() - startNs
    }
}
