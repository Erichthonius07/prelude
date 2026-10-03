package com.prelude.denoise.timeout;

/**
 * Injectable time source for the discard-race mechanism (T7: injectable clock).
 * Method names include units per rule K1.
 */
public interface TimeSource {

    /** Returns the current value of a monotonic nanosecond clock. */
    long nanoTime();

    /** Blocks the current thread for the specified duration in milliseconds. */
    void sleepMs(long millis) throws InterruptedException;
}
