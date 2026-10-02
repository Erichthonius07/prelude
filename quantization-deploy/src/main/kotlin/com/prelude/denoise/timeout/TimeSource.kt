package com.prelude.denoise.timeout

/**
 * Injectable time source for the discard-race mechanism (T7: injectable clock).
 * Method names include units per rule K1.
 */
interface TimeSource {
    /** Returns the current value of a monotonic nanosecond clock. */
    fun nanoTime(): Long

    /** Blocks the current thread for the specified duration in milliseconds. */
    fun sleepMs(millis: Long)
}

/** Default implementation backed by [System.nanoTime] and [Thread.sleep]. */
object SystemTimeSource : TimeSource {
    override fun nanoTime(): Long = System.nanoTime()
    override fun sleepMs(millis: Long) = Thread.sleep(millis)
}
