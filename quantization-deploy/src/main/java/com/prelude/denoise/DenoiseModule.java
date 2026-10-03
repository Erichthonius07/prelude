package com.prelude.denoise;

import com.prelude.denoise.model.DenoisedFrame;
import com.prelude.denoise.model.FusedFrame;
import com.prelude.denoise.model.FrameGeometry;
import com.prelude.denoise.model.LiteRtAdapter;
import com.prelude.denoise.timeout.DiscardRaceRunner;
import com.prelude.denoise.timeout.ErrorListener;
import com.prelude.denoise.timeout.InferenceRunner;
import com.prelude.denoise.timeout.RaceResult;
import com.prelude.denoise.timeout.SystemTimeSource;
import com.prelude.denoise.timeout.TimeSource;
import com.prelude.denoise.tiling.Tiler;
import com.prelude.denoise.tiling.TileInferenceAdapter;
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
 * Both the non-tiled path and the whole-image tiled path run as an
 * InferenceRunner under DiscardRaceRunner, so the timeout (T4, ceiling 500 ms),
 * the single-flight flag (T6) and the fallback semantics (T3/A7/A8) cover both.
 * On a timer win the delivered frame is fusion-only, latencyNs is the
 * time-to-deliver, and the late inference result is discarded, never delivered.
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

    public enum RunMode {
        LIVE,
        BENCHMARK
    }

    private final InferenceRunner inferenceRunner;
    private final TimeSource timeSource;
    private final String modelVariant;
    private final String precision;
    private final ErrorListener errorListener;

    private final AtomicBoolean busyFlag = new AtomicBoolean(false);
    private final AtomicReference<RaceResult> lastRaceResult = new AtomicReference<>();

    private final Tiler tiler;
    private final LiteRtAdapter adapter;

    public DenoiseModule(InferenceRunner inferenceRunner, TimeSource timeSource,
                         String modelVariant, String precision, ErrorListener errorListener,
                         Tiler tiler, LiteRtAdapter adapter) {
        this.inferenceRunner = inferenceRunner;
        this.timeSource = timeSource;
        this.modelVariant = modelVariant;
        this.precision = precision;
        this.errorListener = errorListener;
        this.tiler = tiler;
        this.adapter = adapter;
    }

    public DenoiseModule(InferenceRunner inferenceRunner, TimeSource timeSource,
                         String modelVariant, String precision, ErrorListener errorListener) {
        this(inferenceRunner, timeSource, modelVariant, precision, errorListener, null, null);
    }

    public DenoiseModule(InferenceRunner inferenceRunner, String modelVariant) {
        this(inferenceRunner, SystemTimeSource.INSTANCE, modelVariant, PRECISION_FP32, null);
    }

    /** Non-tiled whole-frame inference under the discard-race. */
    public RaceResult denoise(FusedFrame input, long timeoutMs, RunMode mode) {
        validateTimeoutMs(timeoutMs);
        return denoiseUnderRace(input, timeoutMs, mode, inferenceRunner);
    }

    /**
     * Denoise with explicit geometry (contract gap: FusedFrame has no width/height).
     * The whole-image tiled inference runs as the InferenceRunner under the same
     * discard-race and single-flight core as the non-tiled path.
     */
    public RaceResult denoise(FusedFrame input, FrameGeometry geometry,
                              long timeoutMs, RunMode mode) {
        if (this.tiler == null || this.adapter == null) {
            throw new IllegalStateException("Tiler and LiteRtAdapter must be configured in constructor to use this overload");
        }
        return denoise(input, geometry, timeoutMs, mode, this.tiler, this.adapter);
    }

    /**
     * Denoise with explicit geometry and custom Tiler/LiteRtAdapter.
     */
    public RaceResult denoise(FusedFrame input, FrameGeometry geometry,
                              long timeoutMs, RunMode mode,
                              Tiler tiler, LiteRtAdapter adapter) {
        validateTimeoutMs(timeoutMs);
        TileInferenceAdapter tileAdapter = new TileInferenceAdapter(
            adapter, tiler.getTileSize(), geometry.getChannels());
        InferenceRunner tiledRunner = frame -> tiler.process(
            frame.getImage(), geometry.getWidth(), geometry.getHeight(),
            geometry.getChannels(), tileAdapter);
        return denoiseUnderRace(input, timeoutMs, mode, tiledRunner);
    }

    /**
     * Shared core: single-flight (T6/A26) + discard-race (T1–T3).
     * Identical semantics for the non-tiled and the tiled path; the only
     * difference is which InferenceRunner races against the timer.
     */
    private RaceResult denoiseUnderRace(FusedFrame input, long timeoutMs, RunMode mode,
                                        InferenceRunner raceRunner) {
        if (mode == RunMode.BENCHMARK) {
            RaceResult prev = lastRaceResult.get();
            if (prev != null) {
                try { prev.awaitLateInference(10000L); } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }

        if (!busyFlag.compareAndSet(false, true)) {
            if (mode == RunMode.BENCHMARK) {
                throw new IllegalStateException("Interpreter busy after 10s wait in benchmark mode");
            }
            return RaceResult.busy(new DenoisedFrame(
                input.getBurstId(),
                Arrays.copyOf(input.getImage(), input.getImage().length),
                modelVariant, precision, 0L, false, true
            ));
        }

        try {
            DiscardRaceRunner race = new DiscardRaceRunner(raceRunner, timeSource, errorListener);
            RaceResult result = race.run(
                input, timeoutMs, modelVariant, precision, mode,
                () -> busyFlag.set(false)
            );

            lastRaceResult.set(result);

            if (result.isInferenceWon()) {
                busyFlag.set(false);
            }

            return result;
        } catch (Throwable e) {
            busyFlag.set(false);
            throw e;
        }
    }

    private static void validateTimeoutMs(long timeoutMs) {
        if (timeoutMs < 1 || timeoutMs > DiscardRaceRunner.MAX_TIMEOUT_MS) {
            throw new IllegalArgumentException(
                "Timeout must be 1.." + DiscardRaceRunner.MAX_TIMEOUT_MS + " ms (rule T4)");
        }
    }
}
