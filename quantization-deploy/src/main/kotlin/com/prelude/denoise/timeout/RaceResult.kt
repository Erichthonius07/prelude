package com.prelude.denoise.timeout

import com.prelude.denoise.model.DenoisedFrame
import java.util.concurrent.atomic.AtomicReference

/**
 * Result of a discard-race execution. Holds the delivered frame and exposes
 * the live race state for deferred queries (e.g. [discardRaceOccurred]).
 *
 * The [frame] is the immediately-deliverable result (inference output or
 * fusion-only fallback). The live [state] may transition from
 * [RaceState.TIMER_WON] to [RaceState.DISCARDED] after a late inference
 * completes.
 */
class RaceResult internal constructor(
    /** The frame to deliver downstream (pipeline continues with this). */
    val frame: DenoisedFrame,
    private val state: AtomicReference<RaceState>,
    /**
     * The late inference thread, if the timer won the race.
     * Null when inference won, or when the result is a BUSY fallback.
     */
    val lateInferenceThread: Thread?,
) {
    /** True if inference finished before the timer. */
    val inferenceWon: Boolean get() = state.get() == RaceState.INFERENCE_WON

    /** True if the timer fired before inference completed. */
    val timerWon: Boolean
        get() = state.get().let { it == RaceState.TIMER_WON || it == RaceState.DISCARDED }

    /**
     * True if the timer won AND the late inference later completed,
     * producing a result that was discarded (never delivered).
     * This value may change from false to true after construction.
     */
    val discardRaceOccurred: Boolean get() = state.get() == RaceState.DISCARDED

    /**
     * Wait for a late inference to finish (bounded). Returns the final
     * [discardRaceOccurred] value — suitable for the wire payload.
     *
     * @param maxWaitMs Maximum time to wait in milliseconds.
     */
    fun awaitLateInference(maxWaitMs: Long): Boolean {
        lateInferenceThread?.join(maxWaitMs)
        return discardRaceOccurred
    }

    companion object {
        /** Create a BUSY fallback result (A26/T6). */
        internal fun busy(frame: DenoisedFrame): RaceResult = RaceResult(
            frame = frame,
            state = AtomicReference(RaceState.TIMER_WON),
            lateInferenceThread = null,
        )
    }
}
