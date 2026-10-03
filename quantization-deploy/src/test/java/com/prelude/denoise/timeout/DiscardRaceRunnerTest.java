package com.prelude.denoise.timeout;

import com.prelude.denoise.model.FusedFrame;
import org.junit.Test;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.*;

/**
 * Tests for DiscardRaceRunner covering T7 requirements.
 * Uses injectable clock (FakeTimeSource) and fake inferencer
 * (FakeInferenceRunner) for deterministic results.
 */
public class DiscardRaceRunnerTest {

    private final FusedFrame testInput = new FusedFrame(
        "test-burst-1", "naive",
        new float[]{0.1f, 0.2f, 0.3f}, "fusion_multi"
    );

    // ---- T7 test 1: Timer wins (slow inference) ----

    @Test
    public void timerWinsWhenInferenceIsSlow() throws Exception {
        CountDownLatch inferStarted = new CountDownLatch(1);
        CountDownLatch inferProceed = new CountDownLatch(1);
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(
            new float[]{0.9f}, inferStarted, inferProceed);
        FakeTimeSource fakeTime = new FakeTimeSource();
        DiscardRaceRunner runner = new DiscardRaceRunner(fakeInference, fakeTime, (msg, t) -> {});

        AtomicReference<RaceResult> resultRef = new AtomicReference<>();
        AtomicReference<Throwable> errorRef = new AtomicReference<>();
        CountDownLatch done = new CountDownLatch(1);

        new Thread(() -> {
            try {
                resultRef.set(runner.run(testInput, 100, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.LIVE));
            } catch (Throwable e) {
                errorRef.set(e);
            }
            done.countDown();
        }).start();

        assertTrue("Inference should start", inferStarted.await(5, TimeUnit.SECONDS));
        fakeTime.releaseSleep(); // timer fires; inference still blocked
        assertTrue("run() should complete", done.await(5, TimeUnit.SECONDS));
        assertNull("No error expected", errorRef.get());

        RaceResult result = resultRef.get();
        assertTrue("Timer should have won", result.isTimerWon());
        assertFalse("Inference should not have won", result.isInferenceWon());
        assertTrue("Frame should indicate timeout", result.getFrame().isTimeoutOccurred());
        assertFalse("Discard not yet (inference still running)", result.isDiscardRaceOccurred());
        assertArrayEquals("Fusion-only pixels", testInput.getImage(), result.getFrame().getImage(), 0.0001f);

        // Release inference — should trigger DISCARDED transition
        inferProceed.countDown();
        boolean discarded = result.awaitLateInference(5000);
        assertTrue("Late inference should be discarded", discarded);
        assertTrue("discardRaceOccurred should now be true", result.isDiscardRaceOccurred());
    }

    // ---- T7 test 2: Inference wins (fast inference) ----

    @Test
    public void inferenceWinsWhenFasterThanTimer() {
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(new float[]{0.9f}, 0);
        DiscardRaceRunner runner = new DiscardRaceRunner(fakeInference, SystemTimeSource.INSTANCE, (msg, t) -> {});

        RaceResult result = runner.run(testInput, 500, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.LIVE);

        assertTrue("Inference should have won", result.isInferenceWon());
        assertFalse("Timer should not have won", result.isTimerWon());
        assertFalse("No timeout", result.getFrame().isTimeoutOccurred());
        assertFalse("No discard", result.getFrame().isDiscardRaceOccurred());
        assertArrayEquals("Denoised pixels returned", new float[]{0.9f}, result.getFrame().getImage(), 0.0001f);
        assertTrue("Latency should be recorded", result.getFrame().getLatencyNs() >= 0);
    }

    // ---- T7 test 3: Near-simultaneous ----

    @Test
    public void nearSimultaneousExactlyOneWinner() {
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(new float[]{0.7f}, 5);
        DiscardRaceRunner runner = new DiscardRaceRunner(fakeInference, SystemTimeSource.INSTANCE, (msg, t) -> {});

        RaceResult result = runner.run(testInput, 8, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.LIVE);

        assertTrue("Exactly one winner",
            result.isInferenceWon() ^ result.isTimerWon());
        assertNotNull("Frame always present", result.getFrame());
        assertEquals("BurstId preserved", "test-burst-1", result.getFrame().getBurstId());
    }

    // ---- T7 test 4: Late result never delivered ----

    @Test
    public void lateInferenceResultIsNeverDelivered() throws Exception {
        CountDownLatch inferStarted = new CountDownLatch(1);
        CountDownLatch inferProceed = new CountDownLatch(1);
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(
            new float[]{0.9f}, inferStarted, inferProceed);
        FakeTimeSource fakeTime = new FakeTimeSource();
        DiscardRaceRunner runner = new DiscardRaceRunner(fakeInference, fakeTime, (msg, t) -> {});

        AtomicReference<RaceResult> resultRef = new AtomicReference<>();
        CountDownLatch done = new CountDownLatch(1);
        new Thread(() -> {
            resultRef.set(runner.run(testInput, 50, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.LIVE));
            done.countDown();
        }).start();

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS));
        fakeTime.releaseSleep();
        assertTrue(done.await(5, TimeUnit.SECONDS));

        RaceResult result = resultRef.get();
        assertArrayEquals("Delivered pixels should be fusion-only",
            testInput.getImage(), result.getFrame().getImage(), 0.0001f);

        // Let late inference finish
        inferProceed.countDown();
        result.awaitLateInference(5000);

        // Delivered frame is STILL fusion-only (T3: late result discarded)
        assertArrayEquals("Delivered pixels must not change after late inference",
            testInput.getImage(), result.getFrame().getImage(), 0.0001f);
        assertTrue("discardRaceOccurred should be true", result.isDiscardRaceOccurred());
    }

    // ---- T7 test 5: No double emission ----

    @Test
    public void exactlyOneResultDeliveredNoDoubleEmission() throws Exception {
        AtomicInteger deliveryCount = new AtomicInteger(0);
        CountDownLatch inferStarted = new CountDownLatch(1);
        CountDownLatch inferProceed = new CountDownLatch(1);
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(
            new float[]{0.9f}, inferStarted, inferProceed);
        FakeTimeSource fakeTime = new FakeTimeSource();
        DiscardRaceRunner runner = new DiscardRaceRunner(fakeInference, fakeTime, (msg, t) -> {});

        AtomicReference<RaceResult> resultRef = new AtomicReference<>();
        CountDownLatch done = new CountDownLatch(1);
        new Thread(() -> {
            resultRef.set(runner.run(testInput, 50, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.LIVE));
            deliveryCount.incrementAndGet();
            done.countDown();
        }).start();

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS));
        fakeTime.releaseSleep();
        assertTrue(done.await(5, TimeUnit.SECONDS));
        assertEquals("Exactly one delivery", 1, deliveryCount.get());

        // Let late inference finish — must NOT cause another delivery
        inferProceed.countDown();
        resultRef.get().awaitLateInference(5000);
        assertEquals("Still exactly one delivery", 1, deliveryCount.get());
    }

    // ---- T7 test 6: Interpreter never called concurrently ----

    @Test
    public void inferenceRunnerNeverCalledConcurrently() {
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(new float[]{0.9f}, 50);
        DiscardRaceRunner runner = new DiscardRaceRunner(fakeInference, SystemTimeSource.INSTANCE, (msg, t) -> {});

        runner.run(testInput, 200, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.LIVE);

        assertEquals("Max concurrent calls should be 1 (D4)",
            1, fakeInference.maxConcurrentCalls.get());
    }

    // ---- T7 test 8a: Exception propagates when inference wins the race ----

    @Test
    public void inferenceExceptionYieldsFusionOnlyInLiveMode() throws Exception {
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(
            new RuntimeException("OOM in native"), 0);
        FakeTimeSource fakeTime = new FakeTimeSource();
        DiscardRaceRunner runner = new DiscardRaceRunner(fakeInference, fakeTime, (msg, t) -> {});

        AtomicReference<RaceResult> resultRef = new AtomicReference<>();
        CountDownLatch done = new CountDownLatch(1);

        new Thread(() -> {
            try {
                resultRef.set(runner.run(testInput, 100, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.LIVE));
            } finally {
                done.countDown();
            }
        }).start();

        assertTrue(done.await(5, TimeUnit.SECONDS));
        assertNotNull("Result should be returned", resultRef.get());
        assertTrue("Frame should be fusion-only", resultRef.get().getFrame().isTimeoutOccurred());
        assertEquals("Reason should be inference error", RaceResult.REASON_INFERENCE_ERROR, resultRef.get().getFallbackReason());
    }

    @Test
    public void inferenceExceptionThrowsInBenchmarkMode() throws Exception {
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(
            new RuntimeException("OOM in native"), 0);
        FakeTimeSource fakeTime = new FakeTimeSource();
        DiscardRaceRunner runner = new DiscardRaceRunner(fakeInference, fakeTime, (msg, t) -> {});

        AtomicReference<Throwable> errorRef = new AtomicReference<>();
        CountDownLatch done = new CountDownLatch(1);

        new Thread(() -> {
            try {
                runner.run(testInput, 100, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.BENCHMARK);
            } catch (Throwable e) {
                errorRef.set(e);
            }
            done.countDown();
        }).start();

        assertTrue(done.await(5, TimeUnit.SECONDS));
        assertNotNull("Exception should propagate", errorRef.get());
        assertTrue("Message should contain OOM", errorRef.get().getMessage().contains("OOM in native"));
    }

    // ---- T7 test 8b: Exception swallowed when timer already won ----

    @Test
    public void inferenceExceptionSwallowedWhenTimerAlreadyWon() throws Exception {
        CountDownLatch inferStarted = new CountDownLatch(1);
        CountDownLatch inferProceed = new CountDownLatch(1);
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(
            new RuntimeException("OOM in native"), inferStarted, inferProceed);
        FakeTimeSource fakeTime = new FakeTimeSource();
        CountDownLatch lateCompleteCalled = new CountDownLatch(1);
        DiscardRaceRunner runner = new DiscardRaceRunner(fakeInference, fakeTime, (msg, t) -> {});

        AtomicReference<RaceResult> resultRef = new AtomicReference<>();
        CountDownLatch done = new CountDownLatch(1);
        new Thread(() -> {
            resultRef.set(runner.run(testInput, 50, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.LIVE, lateCompleteCalled::countDown));
            done.countDown();
        }).start();

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS));
        fakeTime.releaseSleep(); // timer wins
        assertTrue(done.await(5, TimeUnit.SECONDS));

        RaceResult result = resultRef.get();
        assertTrue("Timer should have won", result.isTimerWon());

        // Release inference — it will throw, but exception is swallowed
        inferProceed.countDown();
        assertTrue("onLateInferenceComplete should be called even on error",
            lateCompleteCalled.await(5, TimeUnit.SECONDS));
        assertFalse("Failed late inference is not a discard", result.isDiscardRaceOccurred());
    }

    // ---- onLateInferenceComplete callback fires after discard ----

    @Test
    public void onLateInferenceCompleteCallbackFiresAfterDiscard() throws Exception {
        CountDownLatch inferStarted = new CountDownLatch(1);
        CountDownLatch inferProceed = new CountDownLatch(1);
        CountDownLatch callbackFired = new CountDownLatch(1);
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(
            new float[]{0.9f}, inferStarted, inferProceed);
        FakeTimeSource fakeTime = new FakeTimeSource();
        DiscardRaceRunner runner = new DiscardRaceRunner(fakeInference, fakeTime, (msg, t) -> {});

        AtomicReference<RaceResult> resultRef = new AtomicReference<>();
        CountDownLatch done = new CountDownLatch(1);
        new Thread(() -> {
            resultRef.set(runner.run(testInput, 50, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.LIVE, callbackFired::countDown));
            done.countDown();
        }).start();

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS));
        fakeTime.releaseSleep();
        assertTrue(done.await(5, TimeUnit.SECONDS));

        assertEquals("Callback not yet", 1L, callbackFired.getCount());

        inferProceed.countDown();
        assertTrue("Callback should fire after late inference completes",
            callbackFired.await(5, TimeUnit.SECONDS));
    }

    // ---- Timeout bounds (T4) ----

    @Test(expected = IllegalArgumentException.class)
    public void rejectsTimeoutAbove500ms() {
        DiscardRaceRunner runner = new DiscardRaceRunner(
            new FakeInferenceRunner(), SystemTimeSource.INSTANCE, (msg, t) -> {});
        runner.run(testInput, 501, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.LIVE);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsTimeoutOfZero() {
        DiscardRaceRunner runner = new DiscardRaceRunner(
            new FakeInferenceRunner(), SystemTimeSource.INSTANCE, (msg, t) -> {});
        runner.run(testInput, 0, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.LIVE);
    }

    // ---- Contract field validation ----

    @Test
    public void outputCarriesBurstIdUnchanged() {
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(new float[]{0.5f});
        DiscardRaceRunner runner = new DiscardRaceRunner(fakeInference, SystemTimeSource.INSTANCE, (msg, t) -> {});
        RaceResult result = runner.run(testInput, 500, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.LIVE);
        assertEquals("burstId must be carried unchanged (D3)", "test-burst-1", result.getFrame().getBurstId());
    }

    @Test
    public void outputCarriesModelVariantAndPrecision() {
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(new float[]{0.5f});
        DiscardRaceRunner runner = new DiscardRaceRunner(fakeInference, SystemTimeSource.INSTANCE, (msg, t) -> {});
        RaceResult result = runner.run(testInput, 500, "cnn-v1", "int8", com.prelude.denoise.DenoiseModule.RunMode.LIVE);
        assertEquals("cnn-v1", result.getFrame().getModelVariant());
        assertEquals("int8", result.getFrame().getPrecision());
    }

    @Test
    public void latencyMsConversionIsCorrect() {
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(new float[]{0.5f});
        DiscardRaceRunner runner = new DiscardRaceRunner(fakeInference, SystemTimeSource.INSTANCE, (msg, t) -> {});
        RaceResult result = runner.run(testInput, 500, "cnn-v1", "fp32", com.prelude.denoise.DenoiseModule.RunMode.LIVE);
        assertEquals("latencyMs = latencyNs / 1_000_000.0 (D2)",
            result.getFrame().getLatencyNs() / 1_000_000.0,
            result.getFrame().getLatencyMs(), 0.001);
    }
}
