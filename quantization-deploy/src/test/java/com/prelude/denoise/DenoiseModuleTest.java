package com.prelude.denoise;

import com.prelude.denoise.model.FusedFrame;
import com.prelude.denoise.timeout.FakeInferenceRunner;
import com.prelude.denoise.timeout.FakeTimeSource;
import com.prelude.denoise.timeout.InferenceRunner;
import com.prelude.denoise.timeout.RaceResult;
import com.prelude.denoise.timeout.SystemTimeSource;
import org.junit.Test;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.*;

/**
 * Tests for DenoiseModule: single-flight policy (T6/A26),
 * inference thread survival (T1), and pipelineMode isolation (T5).
 */
public class DenoiseModuleTest {

    private FusedFrame input(String burstId) {
        return new FusedFrame(burstId, "naive",
            new float[]{0.1f, 0.2f, 0.3f}, "fusion_multi");
    }

    private FusedFrame input() { return input("burst-1"); }

    // ---- T7 test 7: Repeated bursts / single-flight / BUSY ----

    @Test
    public void singleFlightSecondBurstWhileInferenceIsLateGetsBusyFusionOnly() throws Exception {
        CountDownLatch inferStarted = new CountDownLatch(1);
        CountDownLatch inferProceed = new CountDownLatch(1);
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(
            new float[]{0.9f}, inferStarted, inferProceed);
        FakeTimeSource fakeTime = new FakeTimeSource();
        DenoiseModule module = new DenoiseModule(fakeInference, fakeTime, "cnn-v1", "fp32");

        // First burst — will timeout (inference blocked on proceedLatch)
        AtomicReference<RaceResult> firstRef = new AtomicReference<>();
        CountDownLatch firstDone = new CountDownLatch(1);
        new Thread(() -> {
            firstRef.set(module.denoise(input("burst-1"), 100, DenoiseModule.RunMode.LIVE));
            firstDone.countDown();
        }).start();

        assertTrue("Inference should start", inferStarted.await(5, TimeUnit.SECONDS));
        fakeTime.releaseSleep(); // timer wins; inference still blocked
        assertTrue("First denoise should return", firstDone.await(5, TimeUnit.SECONDS));
        assertTrue("First: timer won", firstRef.get().isTimerWon());

        // Second burst — should get BUSY (inference from burst-1 still running)
        RaceResult secondResult = module.denoise(input("burst-2"), 100, DenoiseModule.RunMode.LIVE);

        assertTrue("Second: timeout (BUSY)", secondResult.getFrame().isTimeoutOccurred());
        assertEquals("Second: burstId preserved (D3)", "burst-2", secondResult.getFrame().getBurstId());
        assertArrayEquals("BUSY frame should have fusion-only pixels",
            input("burst-2").getImage(), secondResult.getFrame().getImage(), 0.0001f);

        // Release first inference — should clear busy flag
        inferProceed.countDown();
        firstRef.get().awaitLateInference(5000);
    }

    // ---- T7 test 9: Inference thread survives after timer wins (T1) ----

    @Test
    public void inferenceThreadContinuesRunningAfterTimerWins() throws Exception {
        CountDownLatch inferStarted = new CountDownLatch(1);
        CountDownLatch inferProceed = new CountDownLatch(1);
        CountDownLatch inferCompleted = new CountDownLatch(1);

        InferenceRunner fakeInference = input1 -> {
            inferStarted.countDown();
            try { inferProceed.await(); } catch (InterruptedException e) { throw new RuntimeException(e); }
            inferCompleted.countDown();
            return new float[]{0.9f};
        };
        FakeTimeSource fakeTime = new FakeTimeSource();
        DenoiseModule module = new DenoiseModule(fakeInference, fakeTime, "cnn-v1", "fp32");

        AtomicReference<RaceResult> resultRef = new AtomicReference<>();
        CountDownLatch done = new CountDownLatch(1);
        new Thread(() -> {
            resultRef.set(module.denoise(input(), 50, DenoiseModule.RunMode.LIVE));
            done.countDown();
        }).start();

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS));
        fakeTime.releaseSleep(); // timer wins
        assertTrue(done.await(5, TimeUnit.SECONDS));

        // run() returned, but inference thread is still alive (T1: never killed)
        Thread lateThread = resultRef.get().getLateInferenceThread();
        assertNotNull("Late inference thread should exist", lateThread);
        assertTrue("Inference thread should still be alive", lateThread.isAlive());

        // Release inference — it should complete naturally
        inferProceed.countDown();
        assertTrue("Inference thread should complete (not killed — T1)",
            inferCompleted.await(5, TimeUnit.SECONDS));
    }

    // ---- T5: pipelineMode does not affect timeout ----

    @Test
    public void pipelineModeDoesNotAffectTimeoutBehavior() {
        FakeInferenceRunner runner1 = new FakeInferenceRunner(new float[]{0.5f});
        DenoiseModule module1 = new DenoiseModule(runner1, SystemTimeSource.INSTANCE, "cnn-v1", "fp32");
        FusedFrame multiInput = new FusedFrame("b1", "naive",
            new float[]{0.1f}, "fusion_multi");
        RaceResult result1 = module1.denoise(multiInput, 500, DenoiseModule.RunMode.LIVE);

        FakeInferenceRunner runner2 = new FakeInferenceRunner(new float[]{0.5f});
        DenoiseModule module2 = new DenoiseModule(runner2, SystemTimeSource.INSTANCE, "cnn-v1", "fp32");
        FusedFrame singleInput = new FusedFrame("b2", "naive",
            new float[]{0.1f}, "fusion_single");
        RaceResult result2 = module2.denoise(singleInput, 500, DenoiseModule.RunMode.LIVE);

        assertEquals("pipelineMode must not affect timeout (T5)",
            result1.isInferenceWon(), result2.isInferenceWon());
    }

    // ---- Timeout bounds ----

    @Test(expected = IllegalArgumentException.class)
    public void rejectsTimeoutAboveCeiling() {
        DenoiseModule module = new DenoiseModule(new FakeInferenceRunner(), "cnn-v1");
        module.denoise(input(), 501, DenoiseModule.RunMode.LIVE);
    }
}
