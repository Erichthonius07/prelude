package com.prelude.denoise.timeout

import com.prelude.denoise.model.DenoisedFrame
import com.prelude.denoise.model.FusedFrame
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

/**
 * Core discard-race mechanism (rules T1–T8).
 *
 * Inference runs on a dedicated thread; a timer runs in parallel.
 * First finisher wins via an atomic compare-and-set. Exactly one
 * result is ever delivered. The inference thread is NEVER killed (T1).
 *
 * @param inferenceRunner The (possibly slow) synchronous inference call.
 * @param timeSource Injectable clock for testing (T7).
 */
class DiscardRaceRunner(
    private val inferenceRunner: InferenceRunner,
    private val timeSource: TimeSource,
) {
    /**
     * Run inference with a competing timer.
     *
     * @param input The fused frame to denoise.
     * @param timeoutMs Calibrated timeout in ms (T4: 1..[MAX_TIMEOUT_MS]).
     * @param modelVariant Model name for the output frame (C6).
     * @param precision "fp32" or "int8".
     * @param onLateInferenceComplete Called when a late inference finishes
     *     (or fails) after the timer already won. Used by [DenoiseModule][com.prelude.denoise.DenoiseModule]
     *     to clear the single-flight busy flag (A26/T6).
     * @return [RaceResult] with the frame and live race state.
     * @throws Throwable if inference fails AND wins the race.
     */
    fun run(
        input: FusedFrame,
        timeoutMs: Long,
        modelVariant: String,
        precision: String,
        onLateInferenceComplete: (() -> Unit)? = null,
    ): RaceResult {
        require(timeoutMs in 1..MAX_TIMEOUT_MS) {
            "Timeout must be 1..$MAX_TIMEOUT_MS ms (rule T4, ceiling enforced)"
        }

        val state = AtomicReference(RaceState.PENDING)
        val frameRef = AtomicReference<DenoisedFrame>()
        val errorRef = AtomicReference<Throwable>()
        val latch = CountDownLatch(1)
        val lateThreadRef = AtomicReference<Thread>()

        // --- Inference thread (T1: synchronous, not cancellable) ---
        val inferThread = thread(name = "denoise-inference", isDaemon = true) {
            try {
                val startNs = timeSource.nanoTime()
                val pixels = inferenceRunner.run(input)
                val elapsedNs = timeSource.nanoTime() - startNs

                if (state.compareAndSet(RaceState.PENDING, RaceState.INFERENCE_WON)) {
                    // T2: inference won the race
                    frameRef.set(
                        DenoisedFrame(
                            burstId = input.burstId,
                            image = pixels,
                            modelVariant = modelVariant,
                            precision = precision,
                            latencyNs = elapsedNs,
                            discardRaceOccurred = false,
                            timeoutOccurred = false,
                        )
                    )
                    latch.countDown()
                } else {
                    // T3: timer already won — result is DISCARDED
                    state.compareAndSet(RaceState.TIMER_WON, RaceState.DISCARDED)
                    onLateInferenceComplete?.invoke()
                }
            } catch (e: Throwable) {
                if (state.compareAndSet(RaceState.PENDING, RaceState.FAILED)) {
                    // Inference failed before timer — propagate to caller
                    errorRef.set(e)
                    latch.countDown()
                } else {
                    // Timer already won and late inference failed — not a discard
                    onLateInferenceComplete?.invoke()
                }
            }
        }

        // --- Timer thread (parallel) ---
        thread(name = "denoise-timer", isDaemon = true) {
            try {
                timeSource.sleepMs(timeoutMs)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
                return@thread
            }

            if (state.compareAndSet(RaceState.PENDING, RaceState.TIMER_WON)) {
                // T3: timer won — emit fusion-only frame immediately
                frameRef.set(
                    DenoisedFrame(
                        burstId = input.burstId,
                        image = input.image.copyOf(), // fusion-only pixels
                        modelVariant = modelVariant,
                        precision = precision,
                        latencyNs = 0L,
                        discardRaceOccurred = false, // may become true via DISCARDED state
                        timeoutOccurred = true,
                    )
                )
                lateThreadRef.set(inferThread)
                latch.countDown()
            }
            // If inference already won or failed, timer is a no-op
        }

        latch.await() // exactly one thread counts down

        // Propagate inference errors if inference won the race with an exception
        errorRef.get()?.let { throw it }

        return RaceResult(
            frame = frameRef.get(),
            state = state,
            lateInferenceThread = lateThreadRef.get(),
        )
    }

    companion object {
        /** Hard ceiling for timeout value (rule T4). */
        const val MAX_TIMEOUT_MS = 500L

        /**
         * Fallback timeout for uncalibrated devices in dev/test builds (T4).
         * Benchmark/graded runs MUST refuse to run with this value.
         */
        const val UNCALIBRATED_TIMEOUT_MS = 500L
    }
}
