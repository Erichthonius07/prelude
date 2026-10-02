package com.prelude.denoise.api

import com.prelude.denoise.model.FusedFrame
import com.prelude.denoise.timeout.InferenceRunner
import com.prelude.denoise.timeout.SystemTimeSource
import com.prelude.denoise.timeout.TimeSource

/**
 * Hook for reading the device's thermal state before/during calibration.
 * Role 1 provides the implementation (e.g. mapping to Android's PowerManager).
 */
interface ThermalStateProvider {
    fun getCurrentThermalStatus(): Int
}

/**
 * Clean API for Role 1's `InferenceTimeoutCalculator` to run real inference
 * and collect per-image latencies (rule D6).
 *
 * This computes the nearest-rank p95 latency.
 * No concurrent calls allowed (rule D4).
 */
class InferenceApi(
    private val inferenceRunner: InferenceRunner,
    private val timeSource: TimeSource = SystemTimeSource,
) {
    /**
     * Executes a calibration run of [runs] inferences (excluding [warmupRuns])
     * on the actual LiteRT inference engine.
     * 
     * @return The nearest-rank p95 latency in milliseconds.
     * @throws IllegalStateException if the thermal state exceeds normal (e.g., > 1) during calibration.
     */
    fun calibrateLatency(
        runs: Int = 100,
        warmupRuns: Int = 5,
        thermalStateProvider: ThermalStateProvider,
        dummyInput: FusedFrame // Required to feed the pipeline
    ): Long {
        if (thermalStateProvider.getCurrentThermalStatus() > 1) { // 1 = THERMAL_STATUS_LIGHT, 0 = NONE
            throw IllegalStateException("Device is thermally throttled before calibration.")
        }

        // Warm-up phase
        for (i in 0 until warmupRuns) {
            inferenceRunner.run(dummyInput)
        }

        val latencies = LongArray(runs)

        // Measurement phase
        for (i in 0 until runs) {
            if (thermalStateProvider.getCurrentThermalStatus() > 1) {
                throw IllegalStateException("Device thermally throttled during calibration.")
            }

            val startNs = timeSource.nanoTime()
            inferenceRunner.run(dummyInput)
            val durationNs = timeSource.nanoTime() - startNs
            latencies[i] = durationNs
        }

        // Nearest-rank p95 computation
        latencies.sort()
        val p95Index = Math.ceil(0.95 * runs).toInt() - 1
        val safeIndex = p95Index.coerceIn(0, runs - 1)
        
        // Convert to milliseconds
        return latencies[safeIndex] / 1_000_000L
    }
}
