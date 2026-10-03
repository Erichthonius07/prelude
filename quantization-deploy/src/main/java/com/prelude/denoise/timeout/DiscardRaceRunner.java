package com.prelude.denoise.timeout;

import com.prelude.denoise.model.DenoisedFrame;
import com.prelude.denoise.model.FusedFrame;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
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

    public static final long MAX_TIMEOUT_MS = 500L;
    public static final long UNCALIBRATED_TIMEOUT_MS = 500L;

    private final InferenceRunner inferenceRunner;
    private final TimeSource timeSource;
    private final ErrorListener errorListener;

    public DiscardRaceRunner(InferenceRunner inferenceRunner, TimeSource timeSource, ErrorListener errorListener) {
        this.inferenceRunner = inferenceRunner;
        this.timeSource = timeSource;
        this.errorListener = errorListener;
    }

    public RaceResult run(FusedFrame input, long timeoutMs, String modelVariant, String precision,
                          com.prelude.denoise.DenoiseModule.RunMode mode) {
        return run(input, timeoutMs, modelVariant, precision, mode, null, null);
    }

    public RaceResult run(FusedFrame input, long timeoutMs, String modelVariant, String precision,
                          com.prelude.denoise.DenoiseModule.RunMode mode, Runnable onLateInferenceComplete) {
        return run(input, timeoutMs, modelVariant, precision, mode, onLateInferenceComplete, null);
    }

    /**
     * @param abandonFlag rule T9: set to true by the timer thread when it wins;
     *        a tiled late run checks it between tiles and stops cooperatively
     *        (its InferenceRunner then returns null). Null = no abandon support
     *        (non-tiled runners).
     */
    public RaceResult run(FusedFrame input, long timeoutMs, String modelVariant, String precision,
                          com.prelude.denoise.DenoiseModule.RunMode mode, Runnable onLateInferenceComplete,
                          AtomicBoolean abandonFlag) {
        if (timeoutMs < 1 || timeoutMs > MAX_TIMEOUT_MS) {
            throw new IllegalArgumentException("Timeout must be 1.." + MAX_TIMEOUT_MS + " ms (rule T4, ceiling enforced)");
        }

        AtomicReference<RaceState> state = new AtomicReference<>(RaceState.PENDING);
        AtomicReference<DenoisedFrame> frameRef = new AtomicReference<>();
        AtomicReference<Throwable> errorRef = new AtomicReference<>();
        AtomicReference<String> fallbackReasonRef = new AtomicReference<>(null);
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<Thread> lateThreadRef = new AtomicReference<>();
        AtomicReference<Thread> timerThreadRef = new AtomicReference<>();

        long raceStartNs = timeSource.nanoTime();

        Thread inferThread = new Thread(() -> {
            try {
                float[] pixels = inferenceRunner.run(input);
                long elapsedNs = timeSource.nanoTime() - raceStartNs;

                if (pixels == null && state.compareAndSet(RaceState.PENDING, RaceState.FAILED)) {
                    // A null result is only legitimate AFTER the timer won (T9 abandon).
                    // Null while the race is still PENDING is a bug -> inference error.
                    if (mode == com.prelude.denoise.DenoiseModule.RunMode.BENCHMARK) {
                        errorRef.set(new IllegalStateException(
                            "Inference returned null while the race was PENDING (bug)"));
                    } else {
                        if (errorListener != null) {
                            errorListener.onError(
                                "Inference returned null while the race was PENDING (bug)", null);
                        }
                        long nullElapsedNs = timeSource.nanoTime() - raceStartNs;
                        frameRef.set(new DenoisedFrame(
                            input.getBurstId(), Arrays.copyOf(input.getImage(), input.getImage().length),
                            modelVariant, precision, nullElapsedNs, false, true
                        ));
                        fallbackReasonRef.set(RaceResult.REASON_INFERENCE_ERROR);
                    }
                    latch.countDown();
                } else if (pixels != null && state.compareAndSet(RaceState.PENDING, RaceState.INFERENCE_WON)) {
                    frameRef.set(new DenoisedFrame(
                        input.getBurstId(), pixels, modelVariant, precision,
                        elapsedNs, false, false
                    ));
                    Thread t = timerThreadRef.get();
                    if (t != null) t.interrupt();
                    latch.countDown();
                } else {
                    // Late completion: the run was abandoned between tiles (T9, null)
                    // or finished/failed after the timer had already won. Result is dropped.
                    state.compareAndSet(RaceState.TIMER_WON, RaceState.DISCARDED);
                    if (onLateInferenceComplete != null) onLateInferenceComplete.run();
                }
            } catch (Throwable e) {
                if (state.compareAndSet(RaceState.PENDING, RaceState.FAILED)) {
                    if (mode == com.prelude.denoise.DenoiseModule.RunMode.BENCHMARK) {
                        errorRef.set(e);
                    } else {
                        if (errorListener != null) {
                            errorListener.onError("Inference failed", e);
                        }
                        long elapsedNs = timeSource.nanoTime() - raceStartNs;
                        frameRef.set(new DenoisedFrame(
                            input.getBurstId(), Arrays.copyOf(input.getImage(), input.getImage().length),
                            modelVariant, precision, elapsedNs, false, true
                        ));
                        fallbackReasonRef.set(RaceResult.REASON_INFERENCE_ERROR);
                    }
                    latch.countDown();
                } else {
                    if (onLateInferenceComplete != null) onLateInferenceComplete.run();
                }
            }
        }, "denoise-inference");
        inferThread.setDaemon(true);

        Thread timerThread = new Thread(() -> {
            try {
                timeSource.sleepMs(timeoutMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }

            if (state.compareAndSet(RaceState.PENDING, RaceState.TIMER_WON)) {
                // T9: tell the late tiled run to stop at the next tile boundary.
                if (abandonFlag != null) {
                    abandonFlag.set(true);
                }
                long elapsedNs = timeSource.nanoTime() - raceStartNs;
                frameRef.set(new DenoisedFrame(
                    input.getBurstId(), Arrays.copyOf(input.getImage(), input.getImage().length),
                    modelVariant, precision, elapsedNs, false, true
                ));
                lateThreadRef.set(inferThread);
                fallbackReasonRef.set(RaceResult.REASON_TIMEOUT);
                latch.countDown();
            }
        }, "denoise-timer");
        timerThread.setDaemon(true);
        timerThreadRef.set(timerThread);

        inferThread.start();
        timerThread.start();

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while waiting for race result", e);
        }

        Throwable err = errorRef.get();
        if (err != null) {
            if (err instanceof RuntimeException) throw (RuntimeException) err;
            throw new RuntimeException(err);
        }

        return new RaceResult(frameRef.get(), state, lateThreadRef.get(), fallbackReasonRef.get());
    }
}
