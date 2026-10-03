package com.prelude.denoise.api;

import com.prelude.denoise.model.FusedFrame;
import com.prelude.denoise.timeout.InferenceRunner;
import com.prelude.denoise.timeout.SystemTimeSource;
import com.prelude.denoise.timeout.TimeSource;
import java.util.Arrays;

/**
 * Clean API for Role 1's InferenceTimeoutCalculator to run real inference
 * and collect per-image latencies (rule D6).
 *
 * This computes the nearest-rank p95 latency.
 * No concurrent calls allowed (rule D4).
 */
public final class InferenceApi {

    private final InferenceRunner inferenceRunner;
    private final TimeSource timeSource;

    public InferenceApi(InferenceRunner inferenceRunner, TimeSource timeSource) {
        this.inferenceRunner = inferenceRunner;
        this.timeSource = timeSource;
    }

    public InferenceApi(InferenceRunner inferenceRunner) {
        this(inferenceRunner, SystemTimeSource.INSTANCE);
    }

    /**
     * Executes a calibration run of 'runs' inferences (excluding 'warmupRuns')
     * on the actual LiteRT inference engine.
     *
     * @param runs                 number of measurement iterations (default 100)
     * @param warmupRuns           number of warm-up iterations excluded from stats (default 5)
     * @param thermalStateProvider hook to check device thermal status
     * @param dummyInput           representative input to feed the pipeline
     * @return the nearest-rank p95 latency in milliseconds
     * @throws IllegalStateException if thermal state exceeds normal during calibration
     */
    public double calibrateLatency(int runs, int warmupRuns,
                                 ThermalStateProvider thermalStateProvider,
                                 FusedFrame dummyInput) {
        if (thermalStateProvider.getCurrentThermalStatus() > 1) {
            throw new IllegalStateException("Device is thermally throttled before calibration.");
        }

        // Warm-up phase
        for (int i = 0; i < warmupRuns; i++) {
            inferenceRunner.run(dummyInput);
        }

        long[] latencies = new long[runs];

        // Measurement phase
        for (int i = 0; i < runs; i++) {
            if (thermalStateProvider.getCurrentThermalStatus() > 1) {
                throw new IllegalStateException("Device thermally throttled during calibration.");
            }
            long startNs = timeSource.nanoTime();
            inferenceRunner.run(dummyInput);
            latencies[i] = timeSource.nanoTime() - startNs;
        }

        // Nearest-rank p95 computation
        Arrays.sort(latencies);
        int p95Index = (int) Math.ceil(0.95 * runs) - 1;
        int safeIndex = Math.max(0, Math.min(p95Index, runs - 1));

        // Convert nanoseconds to milliseconds
        return latencies[safeIndex] / 1_000_000.0;
    }
}
