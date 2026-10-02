package com.prelude.denoise;

import com.prelude.denoise.model.DenoisedFrame;
import com.prelude.denoise.model.FusedFrame;
import com.prelude.denoise.timeout.DiscardRaceRunner;
import com.prelude.denoise.timeout.InferenceRunner;
import com.prelude.denoise.timeout.RaceResult;
import com.prelude.denoise.timeout.SystemTimeSource;
import com.prelude.denoise.timeout.TimeSource;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Top-level denoise module: FusedFrame → DenoisedFrame (§6.3 → §6.4).
 *
 * Responsibilities:
 *   - Run inference with the discard-race timeout (T1–T8)
 *   - Enforce single-flight policy (A26/T6)
 *   - Expose latency measurement API for calibration (D6)
 *
 * This class does NOT read or use FusedFrame.pipelineMode (rules D4/T5).
 *
 * AtomicBoolean works like Python's threading.Lock(blocking=False) —
 * compareAndSet(false, true) is the non-blocking acquire.
 */
public final class DenoiseModule {

    public static final String MODEL_VARIANT_CNN_V1 = "cnn-v1";
    public static final String PRECISION_FP32 = "fp32";
    public static final String PRECISION_INT8 = "int8";

    /** Run mode controls single-flight behavior (A26). */
    public enum RunMode {
        /** New burst while busy → fusion-only (BUSY). */
        LIVE,
        /** New burst while busy → bounded wait, then proceed. */
        BENCHMARK
    }

    private final InferenceRunner inferenceRunner;
    private final TimeSource timeSource;
    private final String modelVariant;
    private final String precision;

    private final AtomicBoolean busyFlag = new AtomicBoolean(false);
    private final AtomicReference<RaceResult> lastRaceResult = new AtomicReference<>();
    private final DiscardRaceRunner runner;

    public DenoiseModule(InferenceRunner inferenceRunner, TimeSource timeSource,
                         String modelVariant, String precision) {
        this.inferenceRunner = inferenceRunner;
        this.timeSource = timeSource;
        this.modelVariant = modelVariant;
        this.precision = precision;
        this.runner = new DiscardRaceRunner(inferenceRunner, timeSource);
    }

    public DenoiseModule(InferenceRunner inferenceRunner, String modelVariant) {
        this(inferenceRunner, SystemTimeSource.INSTANCE, modelVariant, PRECISION_FP32);
    }

    /**
     * Denoise a fused frame with the discard-race timeout.
     *
     * This method intentionally does NOT read FusedFrame.pipelineMode
     * (rules D4/T5 — pipeline_mode must NEVER influence the timeout).
     *
     * @param input     the fused frame (§6.3)
     * @param timeoutMs per-device calibrated timeout (T4), must be 1..500
     * @param mode      LIVE or BENCHMARK (A26)
     * @return RaceResult containing the frame and race outcome metadata
     */
    public RaceResult denoise(FusedFrame input, long timeoutMs, RunMode mode) {
        if (timeoutMs < 1 || timeoutMs > DiscardRaceRunner.MAX_TIMEOUT_MS) {
            throw new IllegalArgumentException(
                "Timeout must be 1.." + DiscardRaceRunner.MAX_TIMEOUT_MS + " ms (rule T4)");
        }

        // A26: benchmark mode — bounded wait for previous late inference
        if (mode == RunMode.BENCHMARK) {
            RaceResult prev = lastRaceResult.get();
            if (prev != null) {
                try { prev.awaitLateInference(timeoutMs); } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }

        // T6: single-flight — if a previous inference is still running,
        // the interpreter must not be used concurrently (D4).
        if (!busyFlag.compareAndSet(false, true)) {
            // A26: live mode — emit fusion-only with reason BUSY
            return RaceResult.busy(new DenoisedFrame(
                input.getBurstId(),
                Arrays.copyOf(input.getImage(), input.getImage().length),
                modelVariant, precision, 0L, false, true
            ));
        }

        try {
            RaceResult result = runner.run(
                input, timeoutMs, modelVariant, precision,
                () -> busyFlag.set(false)   // onLateInferenceComplete
            );

            lastRaceResult.set(result);

            if (result.isInferenceWon()) {
                busyFlag.set(false);
            }
            // If timer won, busyFlag stays true until onLateInferenceComplete fires

            return result;
        } catch (Throwable e) {
            busyFlag.set(false);
            throw e;
        }
    }
}
