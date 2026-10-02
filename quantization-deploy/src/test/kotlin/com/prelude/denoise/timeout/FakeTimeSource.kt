package com.prelude.denoise.timeout

import java.util.concurrent.CountDownLatch
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Injectable time source for deterministic testing (rule T7).
 *
 * [sleepMs] blocks until [releaseSleep] is called, giving the test full
 * control over when the timer fires. [nanoTime] returns a manually
 * advanced clock.
 */
class FakeTimeSource : TimeSource {

    private val clock = AtomicLong(0L)
    private val sleepQueue = LinkedBlockingQueue<CountDownLatch>()

    /** How many times [sleepMs] has been called. */
    val sleepCount = AtomicInteger(0)

    override fun nanoTime(): Long = clock.get()

    override fun sleepMs(millis: Long) {
        sleepCount.incrementAndGet()
        val gate = CountDownLatch(1)
        sleepQueue.put(gate)
        gate.await()
        // Advance the clock by the sleep duration
        clock.addAndGet(millis * 1_000_000L)
    }

    /**
     * Release the oldest blocked [sleepMs] call (FIFO).
     * Blocks up to 5 seconds for a pending sleep to appear.
     */
    fun releaseSleep() {
        val gate = sleepQueue.poll(5, TimeUnit.SECONDS)
            ?: error("No sleepMs() call to release within 5 seconds")
        gate.countDown()
    }

    /** Manually advance the clock by [nanos] nanoseconds. */
    fun advance(nanos: Long) {
        clock.addAndGet(nanos)
    }
}
