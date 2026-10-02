package com.prelude.denoise.timeout;

import com.prelude.denoise.model.DenoisedFrame;
import com.prelude.denoise.model.FusedFrame;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Core discard-race mechanism (rules T1–T8).
 *
 * Inference runs on a dedicated thread; a timer runs in parallel.
 * First finisher wins via an atomic compare-and-set (CAS). Exactly one
 * result is ever delivered. The inference thread is NEVER killed (T1).
 *
 * Think of AtomicReference as a thread-safe variable that lets two threads
 * race to write to it: only the first compareAndSet(PENDING, X) succeeds.
 * CountDownLatch is like Python's threading.Event() — it blocks until
 * someone calls countDown().
 */
public final class DiscardRaceRunner {

    /** Hard ceiling for timeout value (rule T4). */
    public static final long MAX_TIMEOUT_MS = 500L;

    /** Fallback timeout for uncalibrated devices in dev/test builds (T4). */
    public static final long UNCALIBRATED_TIMEOUT_MS = 500L;

    private final InferenceRunner inferenceRunner;
    private final TimeSource timeSource;

    public DiscardRaceRunner(InferenceRunner inferenceRunner, TimeSource timeSource) {
        this.inferenceRunner = inferenceRunner;
        this.timeSource = timeSource;
    }

    /**
     * Run inference with a competing timer.
     *
     * @param input       the fused frame to denoise
     * @param timeoutMs   calibrated timeout in ms (T4: 1..MAX_TIMEOUT_MS)
     * @param modelVariant model name for the output frame (C6)
     * @param precision   "fp32" or "int8"
     * @return RaceResult with the frame and live race state
     */
    public RaceResult run(FusedFrame input, long timeoutMs,
                          String modelVariant, String precision) {
        return run(input, timeoutMs, modelVariant, precision, null);
    }

    /**
     * Run inference with a competing timer.
     *
     * @param onLateInferenceComplete called when a late inference finishes
     *     (or fails) after the timer already won. Used by DenoiseModule
     *     to clear the single-flight busy flag (A26/T6).
     */
    public RaceResult run(FusedFrame input, long timeoutMs,
                          String modelVariant, String precision,
                          Runnable onLateInferenceComplete) {
        if (timeoutMs < 1 || timeoutMs > MAX_TIMEOUT_MS) {
            throw new IllegalArgumentException(
                "Timeout must be 1.." + MAX_TIMEOUT_MS + " ms (rule T4, ceiling enforced)");
        }

        // Shared state: only ONE thread can move this from PENDING
        AtomicReference<RaceState> state = new AtomicReference<>(RaceState.PENDING);
        AtomicReference<DenoisedFrame> frameRef = new AtomicReference<>();
        AtomicReference<Throwable> errorRef = new AtomicReference<>();
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<Thread> lateThreadRef = new AtomicReference<>();

        // --- Inference thread (T1: synchronous, not cancellable) ---
        Thread inferThread = new Thread(() -> {
            try {
                long startNs = timeSource.nanoTime();
                float[] pixels = inferenceRunner.run(input);
                long elapsedNs = timeSource.nanoTime() - startNs;

                if (state.compareAndSet(RaceState.PENDING, RaceState.INFERENCE_WON)) {
                    // T2: inference won the race
                    frameRef.set(new DenoisedFrame(
                        input.getBurstId(), pixels, modelVariant, precision,
                        elapsedNs, false, false
                    ));
                    latch.countDown();
                } else {
                    // T3: timer already won — result is DISCARDED
                    state.compareAndSet(RaceState.TIMER_WON, RaceState.DISCARDED);
                    if (onLateInferenceComplete != null) onLateInferenceComplete.run();
                }
            } catch (Throwable e) {
                if (state.compareAndSet(RaceState.PENDING, RaceState.FAILED)) {
                    // Inference failed before timer — propagate to caller
                    errorRef.set(e);
                    latch.countDown();
                } else {
                    // Timer already won and late inference failed
                    if (onLateInferenceComplete != null) onLateInferenceComplete.run();
                }
            }
        }, "denoise-inference");
        inferThread.setDaemon(true);
        inferThread.start();

        // --- Timer thread (parallel) ---
        Thread timerThread = new Thread(() -> {
            try {
                timeSource.sleepMs(timeoutMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }

            if (state.compareAndSet(RaceState.PENDING, RaceState.TIMER_WON)) {
                // T3: timer won — emit fusion-only frame immediately
                frameRef.set(new DenoisedFrame(
                    input.getBurstId(),
                    Arrays.copyOf(input.getImage(), input.getImage().length),
                    modelVariant, precision,
                    0L, false, true
                ));
                lateThreadRef.set(inferThread);
                latch.countDown();
            }
            // If inference already won or failed, timer is a no-op
        }, "denoise-timer");
        timerThread.setDaemon(true);
        timerThread.start();

        try {
            latch.await();  // exactly one thread counts down
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while waiting for race result", e);
        }

        // Propagate inference errors if inference won the race with an exception
        Throwable err = errorRef.get();
        if (err != null) {
            if (err instanceof RuntimeException) throw (RuntimeException) err;
            throw new RuntimeException(err);
        }

        return new RaceResult(frameRef.get(), state, lateThreadRef.get());
    }
}
