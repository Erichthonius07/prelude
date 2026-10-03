package com.prelude.denoise.timeout;

/**
 * State machine for the discard-race mechanism (rules T1–T3).
 *
 * Transitions (all via compare-and-set on AtomicReference, exactly one succeeds):
 *   PENDING → INFERENCE_WON : inference finished first
 *   PENDING → TIMER_WON     : timer fired first
 *   PENDING → FAILED        : inference threw an exception before timer
 *   TIMER_WON → DISCARDED   : late inference completed after timer won
 */
public enum RaceState {

    /** Initial state — race in progress. */
    PENDING,

    /** Inference finished before the timer. */
    INFERENCE_WON,

    /** Timer fired before inference completed. */
    TIMER_WON,

    /** Timer won AND late inference later completed — result was dropped. */
    DISCARDED,

    /** Inference threw an exception before the timer fired. */
    FAILED
}
