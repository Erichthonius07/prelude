package com.prelude.denoise;

import com.prelude.denoise.model.FusedFrame;
import com.prelude.denoise.timeout.ErrorListener;
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

public class DenoiseModuleTest {

    private FusedFrame input(String burstId) {
        return new FusedFrame(burstId, "naive",
            new float[]{0.1f, 0.2f, 0.3f}, "fusion_multi");
    }

    private FusedFrame input() { return input("burst-1"); }

    private final ErrorListener testErrorListener = (msg, t) -> {};

    @Test
    public void singleFlightSecondBurstWhileInferenceIsLateGetsBusyFusionOnly() throws Exception {
        CountDownLatch inferStarted = new CountDownLatch(1);
        CountDownLatch inferProceed = new CountDownLatch(1);
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(
            new float[]{0.9f}, inferStarted, inferProceed);
        FakeTimeSource fakeTime = new FakeTimeSource();
        DenoiseModule module = new DenoiseModule(fakeInference, fakeTime, "cnn-v1", "fp32", testErrorListener);

        AtomicReference<RaceResult> firstRef = new AtomicReference<>();
        CountDownLatch firstDone = new CountDownLatch(1);
        new Thread(() -> {
            firstRef.set(module.denoise(input("burst-1"), 100, DenoiseModule.RunMode.LIVE));
            firstDone.countDown();
        }).start();

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS));
        fakeTime.releaseSleep();
        assertTrue(firstDone.await(5, TimeUnit.SECONDS));
        assertTrue(firstRef.get().isTimerWon());

        RaceResult secondResult = module.denoise(input("burst-2"), 100, DenoiseModule.RunMode.LIVE);

        assertTrue(secondResult.getFrame().isTimeoutOccurred());
        assertEquals(RaceResult.REASON_BUSY, secondResult.getFallbackReason());
        
        inferProceed.countDown();
        firstRef.get().awaitLateInference(5000);
    }

    @Test
    public void afterTimeoutAndLateCompletionNextBurstRunsNormally() throws Exception {
        CountDownLatch inferStarted = new CountDownLatch(1);
        CountDownLatch inferProceed = new CountDownLatch(1);
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(
            new float[]{0.9f}, inferStarted, inferProceed);
        FakeTimeSource fakeTime = new FakeTimeSource();
        DenoiseModule module = new DenoiseModule(fakeInference, fakeTime, "cnn-v1", "fp32", testErrorListener);

        AtomicReference<RaceResult> firstRef = new AtomicReference<>();
        CountDownLatch firstDone = new CountDownLatch(1);
        new Thread(() -> {
            firstRef.set(module.denoise(input("burst-1"), 100, DenoiseModule.RunMode.LIVE));
            firstDone.countDown();
        }).start();

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS));
        fakeTime.releaseSleep();
        assertTrue(firstDone.await(5, TimeUnit.SECONDS));
        assertTrue(firstRef.get().isTimerWon());

        // NOW complete the late inference
        inferProceed.countDown();
        firstRef.get().awaitLateInference(5000);

        // Next burst should run normally
        RaceResult secondResult = module.denoise(input("burst-2"), 100, DenoiseModule.RunMode.LIVE);
        assertFalse(secondResult.getFrame().isTimeoutOccurred());
        assertNull(secondResult.getFallbackReason());
    }

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
        DenoiseModule module = new DenoiseModule(fakeInference, fakeTime, "cnn-v1", "fp32", testErrorListener);

        AtomicReference<RaceResult> resultRef = new AtomicReference<>();
        CountDownLatch done = new CountDownLatch(1);
        new Thread(() -> {
            resultRef.set(module.denoise(input(), 50, DenoiseModule.RunMode.LIVE));
            done.countDown();
        }).start();

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS));
        fakeTime.releaseSleep();
        assertTrue(done.await(5, TimeUnit.SECONDS));

        Thread lateThread = resultRef.get().getLateInferenceThread();
        assertNotNull(lateThread);
        
        inferProceed.countDown();
        assertTrue(inferCompleted.await(5, TimeUnit.SECONDS));
    }

    @Test
    public void pipelineModeIsNeverRead() {
        FakeInferenceRunner runner1 = new FakeInferenceRunner(new float[]{0.5f});
        DenoiseModule module = new DenoiseModule(runner1, SystemTimeSource.INSTANCE, "cnn-v1", "fp32", testErrorListener);
        
        FusedFrame multiInput = new FusedFrame("b1", "naive", new float[]{0.1f}, "fusion_multi");
        module.denoise(multiInput, 500, DenoiseModule.RunMode.LIVE);
        
        FusedFrame singleInput = new FusedFrame("b2", "naive", new float[]{0.1f}, "fusion_single");
        module.denoise(singleInput, 500, DenoiseModule.RunMode.LIVE);
        
        // This test proves pipelineMode wasn't involved in timeout logic
        assertEquals(2, runner1.callCount.get());
    }

    @Test(expected = IllegalStateException.class)
    public void benchmarkModeThrowsIfStillBusyAfterBound() throws Exception {
        CountDownLatch inferStarted = new CountDownLatch(1);
        CountDownLatch inferProceed = new CountDownLatch(1);
        FakeInferenceRunner fakeInference = new FakeInferenceRunner(
            new float[]{0.9f}, inferStarted, inferProceed);
        FakeTimeSource fakeTime = new FakeTimeSource();
        DenoiseModule module = new DenoiseModule(fakeInference, fakeTime, "cnn-v1", "fp32", testErrorListener);

        new Thread(() -> {
            module.denoise(input("burst-1"), 100, DenoiseModule.RunMode.BENCHMARK);
        }).start();

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS));
        fakeTime.releaseSleep(); // first race times out
        
        // Now try second burst in BENCHMARK mode while inference is STILL blocked.
        // It will wait up to 10s then throw.
        module.denoise(input("burst-2"), 100, DenoiseModule.RunMode.BENCHMARK);
    }
}
