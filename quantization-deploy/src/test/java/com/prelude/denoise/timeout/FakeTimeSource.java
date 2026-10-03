package com.prelude.denoise.timeout;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Injectable time source for deterministic testing (rule T7).
 *
 * sleepMs() blocks until releaseSleep() is called, giving the test full
 * control over when the timer fires. nanoTime() returns a manually
 * advanced clock.
 */
public class FakeTimeSource implements TimeSource {

    private final AtomicLong clock = new AtomicLong(0L);
    private final LinkedBlockingQueue<CountDownLatch> sleepQueue = new LinkedBlockingQueue<>();

    /** How many times sleepMs has been called. */
    public final AtomicInteger sleepCount = new AtomicInteger(0);

    @Override
    public long nanoTime() { return clock.get(); }

    @Override
    public void sleepMs(long millis) throws InterruptedException {
        sleepCount.incrementAndGet();
        CountDownLatch gate = new CountDownLatch(1);
        sleepQueue.put(gate);
        gate.await();
        // Advance the clock by the sleep duration
        clock.addAndGet(millis * 1_000_000L);
    }

    /**
     * Release the oldest blocked sleepMs call (FIFO).
     * Blocks up to 5 seconds for a pending sleep to appear.
     */
    public void releaseSleep() {
        CountDownLatch gate;
        try {
            gate = sleepQueue.poll(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        if (gate == null) {
            throw new IllegalStateException("No sleepMs() call to release within 5 seconds");
        }
        gate.countDown();
    }

    /** Manually advance the clock by the given nanoseconds. */
    public void advance(long nanos) {
        clock.addAndGet(nanos);
    }
}
