package com.prelude.denoise.timeout;

/**
 * Default implementation backed by System.nanoTime() and Thread.sleep().
 * Singleton — use SystemTimeSource.INSTANCE.
 */
public final class SystemTimeSource implements TimeSource {

    public static final SystemTimeSource INSTANCE = new SystemTimeSource();

    private SystemTimeSource() { }

    @Override
    public long nanoTime() { return System.nanoTime(); }

    @Override
    public void sleepMs(long millis) throws InterruptedException {
        Thread.sleep(millis);
    }
}
