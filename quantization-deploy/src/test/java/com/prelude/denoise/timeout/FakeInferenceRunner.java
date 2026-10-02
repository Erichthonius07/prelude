package com.prelude.denoise.timeout;

import com.prelude.denoise.model.FusedFrame;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Test double for InferenceRunner. Provides latch-based synchronization
 * for deterministic race-condition testing (T7).
 *
 * result       — the pixel data to return from run()
 * delayMs      — simulated inference delay (real Thread.sleep)
 * startedLatch — counted down when run() is entered (before any delay)
 * proceedLatch — run() blocks on this after startedLatch fires;
 *                release it from the test to let inference complete
 * shouldThrow  — if non-null, thrown after delay/proceed
 */
public class FakeInferenceRunner implements InferenceRunner {

    private final float[] result;
    private final long delayMs;
    private final CountDownLatch startedLatch;
    private final CountDownLatch proceedLatch;
    private final Exception shouldThrow;

    /** Total number of run() calls completed (including exceptions). */
    public final AtomicInteger callCount = new AtomicInteger(0);

    /** Number of run() calls currently in flight. */
    public final AtomicInteger concurrentCalls = new AtomicInteger(0);

    /** High-water mark of concurrentCalls. Should be 1 (rule D4). */
    public final AtomicInteger maxConcurrentCalls = new AtomicInteger(0);

    public FakeInferenceRunner(float[] result, long delayMs,
                               CountDownLatch startedLatch,
                               CountDownLatch proceedLatch,
                               Exception shouldThrow) {
        this.result = result;
        this.delayMs = delayMs;
        this.startedLatch = startedLatch;
        this.proceedLatch = proceedLatch;
        this.shouldThrow = shouldThrow;
    }

    public FakeInferenceRunner(float[] result) {
        this(result, 0, null, null, null);
    }

    public FakeInferenceRunner(float[] result, long delayMs) {
        this(result, delayMs, null, null, null);
    }

    public FakeInferenceRunner(float[] result, CountDownLatch startedLatch,
                               CountDownLatch proceedLatch) {
        this(result, 0, startedLatch, proceedLatch, null);
    }

    public FakeInferenceRunner(Exception shouldThrow, long delayMs) {
        this(new float[]{0.5f}, delayMs, null, null, shouldThrow);
    }

    public FakeInferenceRunner(Exception shouldThrow,
                               CountDownLatch startedLatch,
                               CountDownLatch proceedLatch) {
        this(new float[]{0.5f}, 0, startedLatch, proceedLatch, shouldThrow);
    }

    public FakeInferenceRunner() {
        this(new float[]{0.5f});
    }

    @Override
    public float[] run(FusedFrame input) {
        int current = concurrentCalls.incrementAndGet();
        maxConcurrentCalls.updateAndGet(prev -> Math.max(prev, current));
        try {
            if (startedLatch != null) startedLatch.countDown();
            if (proceedLatch != null) {
                try { proceedLatch.await(); } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            if (delayMs > 0) {
                try { Thread.sleep(delayMs); } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            if (shouldThrow != null) {
                if (shouldThrow instanceof RuntimeException) throw (RuntimeException) shouldThrow;
                throw new RuntimeException(shouldThrow);
            }
            return result;
        } finally {
            concurrentCalls.decrementAndGet();
            callCount.incrementAndGet();
        }
    }
}
