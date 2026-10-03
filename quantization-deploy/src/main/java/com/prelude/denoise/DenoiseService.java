package com.prelude.denoise;

import com.prelude.denoise.model.DenoisedFrame;
import com.prelude.denoise.model.FusedFrame;
import com.prelude.denoise.model.FrameGeometry;
import com.prelude.denoise.model.LiteRtAdapter;
import com.prelude.denoise.model.LiteRtAdapterImpl;
import com.prelude.denoise.model.ModelSource;
import com.prelude.denoise.timeout.DiscardRaceRunner;
import com.prelude.denoise.timeout.ErrorListener;
import com.prelude.denoise.timeout.RaceResult;
import com.prelude.denoise.timeout.SystemTimeSource;
import com.prelude.denoise.timeout.TimeSource;
import com.prelude.denoise.tiling.Tiler;
import java.nio.ByteBuffer;

/**
 * Deployment entry point for Role 1 / Role 3: builds the tiled denoise path
 * (Tiler 256 / halo 17 / blend 16, LiteRtAdapterImpl from a ModelSource,
 * DenoiseModule) and exposes one denoise call.
 *
 * <p>INPUT DOMAIN (rule C3, pending): the currently bundled checkpoint was
 * trained on sRGB-domain patches, while contract §6.3 defines FusedFrame.image
 * as a LINEAR-domain buffer. This class performs NO conversion (no gamma, no
 * scaling) — feeding a linear FusedFrame to an sRGB model is a known,
 * documented mismatch until Role 4 ships the linear-domain checkpoint. Do not
 * insert conversions here to hide it.
 *
 * <p>Timeout (rule T4): the timeout budget is per-device calibrated
 * {@code DeviceThresholds.inferenceTimeoutMs} (p95 + 20%, ceiling 500 ms).
 * Until a calibrated profile exists, callers may use
 * {@link #UNCALIBRATED_TIMEOUT_MS} (the 500 ms ceiling) via
 * {@link #denoise(FusedFrame, FrameGeometry, RunMode)}, which logs loudly.
 * Graded/benchmark runs must not use the uncalibrated default.
 */
public final class DenoiseService {

    /** Uncalibrated default: the T4 hard ceiling. Loud-logged when used. */
    public static final long UNCALIBRATED_TIMEOUT_MS = DiscardRaceRunner.MAX_TIMEOUT_MS;

    private final LiteRtAdapter adapter;
    private final DenoiseModule module;
    private final ErrorListener errorListener;

    /** Production constructor: model loaded from the given source (asset or file). */
    public DenoiseService(ModelSource modelSource, ErrorListener errorListener) {
        this(modelSource, null, SystemTimeSource.INSTANCE, errorListener);
    }

    /**
     * Test/advanced constructor: explicit adapter and time source. The model
     * buffer is still loaded from the source (single load; also validates the
     * source), but inference goes through the given adapter.
     */
    DenoiseService(ModelSource modelSource, LiteRtAdapter adapter,
                   TimeSource timeSource, ErrorListener errorListener) {
        if (modelSource == null || timeSource == null) {
            throw new IllegalArgumentException("modelSource and timeSource must be non-null");
        }
        this.errorListener = errorListener;
        ByteBuffer modelBuffer = requireBuffer(modelSource.loadModel());
        this.adapter = adapter != null ? adapter : new LiteRtAdapterImpl(modelBuffer, errorListener);
        this.module = new DenoiseModule(
            frame -> {
                throw new UnsupportedOperationException(
                    "Use denoise(FusedFrame, FrameGeometry, ...): the whole-image tiled path "
                        + "is the deployment path; there is no single-call whole-frame runner here");
            },
            timeSource,
            DenoiseModule.MODEL_VARIANT_CNN_V1,
            DenoiseModule.PRECISION_INT8,
            errorListener,
            new Tiler(),
            this.adapter);
    }

    /**
     * Denoise one fused frame through the whole-image tiled path under the
     * discard-race (timeout T4, single-flight T6, T9 abandon). Returns the
     * RaceResult; call {@code getFrame()} for the delivered DenoisedFrame and
     * {@code awaitLateInference(boundedMs)} before flushing wire rows (T3).
     */
    public RaceResult denoise(FusedFrame frame, FrameGeometry geometry,
                              long timeoutMs, DenoiseModule.RunMode mode) {
        return module.denoise(frame, geometry, timeoutMs, mode);
    }

    /**
     * Convenience overload using {@link #UNCALIBRATED_TIMEOUT_MS} (the 500 ms
     * ceiling). Logs loudly; graded/benchmark runs must use the explicit
     * calibrated timeout instead (rule T4).
     */
    public RaceResult denoise(FusedFrame frame, FrameGeometry geometry,
                              DenoiseModule.RunMode mode) {
        if (errorListener != null) {
            errorListener.onInfo("UNCALIBRATED timeout: using the " + UNCALIBRATED_TIMEOUT_MS
                + " ms ceiling (rule T4). Calibrate the device before graded runs.");
        }
        return denoise(frame, geometry, UNCALIBRATED_TIMEOUT_MS, mode);
    }

    /** Closes the underlying interpreter (never mid-call; rule D4). */
    public void close() {
        adapter.close();
    }

    private static ByteBuffer requireBuffer(ByteBuffer buffer) {
        if (buffer == null || !buffer.isDirect() || buffer.remaining() == 0) {
            throw new IllegalArgumentException(
                "model source must yield a non-empty direct ByteBuffer");
        }
        return buffer;
    }
}
