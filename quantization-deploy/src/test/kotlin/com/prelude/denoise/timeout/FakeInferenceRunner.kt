package com.prelude.denoise.timeout

import com.prelude.denoise.model.FusedFrame
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max

/**
 * Test double for [InferenceRunner]. Provides latch-based synchronization
 * for deterministic race-condition testing (T7).
 *
 * @param result The pixel data to return from [run].
 * @param delayMs Simulated inference delay (real [Thread.sleep]).
 * @param startedLatch Counted down when [run] is entered (before any delay).
 * @param proceedLatch [run] blocks on this after [startedLatch] fires.
 *     Release it from the test to let inference complete.
 * @param shouldThrow If non-null, thrown after delay/proceed.
 */
class FakeInferenceRunner(
    private val result: FloatArray = floatArrayOf(0.5f),
    private val delayMs: Long = 0,
    private val startedLatch: CountDownLatch? = null,
    private val proceedLatch: CountDownLatch? = null,
    private val shouldThrow: Exception? = null,
) : InferenceRunner {

    /** Total number of [run] calls completed (including exceptions). */
    val callCount = AtomicInteger(0)

    /** Number of [run] calls currently in flight. */
    val concurrentCalls = AtomicInteger(0)

    /** High-water mark of [concurrentCalls]. Should be 1 (rule D4). */
    val maxConcurrentCalls = AtomicInteger(0)

    override fun run(input: FusedFrame): FloatArray {
        val current = concurrentCalls.incrementAndGet()
        maxConcurrentCalls.updateAndGet { max(it, current) }
        try {
            startedLatch?.countDown()
            proceedLatch?.await()
            if (delayMs > 0) Thread.sleep(delayMs)
            shouldThrow?.let { throw it }
            return result
        } finally {
            concurrentCalls.decrementAndGet()
            callCount.incrementAndGet()
        }
    }
}
