package com.prelude.denoise.timeout;

import com.prelude.denoise.model.DenoisedFrame;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Result of a discard-race execution. Holds the delivered frame and exposes
 * the live race state for deferred queries (e.g. discardRaceOccurred).
 *
 * The frame is the immediately-deliverable result (inference output or
 * fusion-only fallback). The live state may transition from
 * TIMER_WON to DISCARDED after a late inference completes.
 */
public final class RaceResult {

    public static final String REASON_BUSY = "BUSY";
    public static final String REASON_TIMEOUT = "TIMEOUT";
    public static final String REASON_INFERENCE_ERROR = "INFERENCE_ERROR";

    /** The frame to deliver downstream (pipeline continues with this). */
    private final DenoisedFrame frame;
    private final AtomicReference<RaceState> state;

    /**
     * The late inference thread, if the timer won the race.
     * Null when inference won, or when the result is a BUSY fallback.
     */
    private final Thread lateInferenceThread;
    
    private final String fallbackReason;

    RaceResult(DenoisedFrame frame, AtomicReference<RaceState> state,
               Thread lateInferenceThread, String fallbackReason) {
        this.frame = frame;
        this.state = state;
        this.lateInferenceThread = lateInferenceThread;
        this.fallbackReason = fallbackReason;
    }

    public DenoisedFrame getFrame() { return frame; }
    public Thread getLateInferenceThread() { return lateInferenceThread; }
    public String getFallbackReason() { return fallbackReason; }

    /** True if inference finished before the timer. */
    public boolean isInferenceWon() {
        return state.get() == RaceState.INFERENCE_WON;
    }

    /** True if the timer fired before inference completed. */
    public boolean isTimerWon() {
        RaceState s = state.get();
        return s == RaceState.TIMER_WON || s == RaceState.DISCARDED;
    }

    /**
     * True if the timer won AND the late inference later completed,
     * producing a result that was discarded (never delivered).
     * This value may change from false to true after construction.
     */
    public boolean isDiscardRaceOccurred() {
        return state.get() == RaceState.DISCARDED;
    }

    /**
     * Wait for a late inference to finish (bounded). Returns the final
     * discardRaceOccurred value — suitable for the wire payload.
     *
     * @param maxWaitMs maximum time to wait in milliseconds.
     * @return whether the late inference was discarded.
     */
    public boolean awaitLateInference(long maxWaitMs) throws InterruptedException {
        if (lateInferenceThread != null) {
            lateInferenceThread.join(maxWaitMs);
        }
        return isDiscardRaceOccurred();
    }

    /** Create a BUSY fallback result (A26/T6). */
    public static RaceResult busy(DenoisedFrame frame) {
        return new RaceResult(
            frame,
            new AtomicReference<>(RaceState.TIMER_WON),
            null,
            REASON_BUSY
        );
    }
}
