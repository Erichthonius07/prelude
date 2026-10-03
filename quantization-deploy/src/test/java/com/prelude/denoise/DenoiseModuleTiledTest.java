package com.prelude.denoise;

import com.prelude.denoise.model.DenoisedFrame;
import com.prelude.denoise.model.FusedFrame;
import com.prelude.denoise.model.FrameGeometry;
import com.prelude.denoise.model.LiteRtAdapter;
import com.prelude.denoise.timeout.ErrorListener;
import com.prelude.denoise.timeout.FakeTimeSource;
import com.prelude.denoise.timeout.RaceResult;
import com.prelude.denoise.tiling.Tiler;
import org.junit.Test;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.*;

/**
 * Whole-image tiled path under the discard-race (Task 1).
 *
 * The tiled overload must run tiler.process as the InferenceRunner inside
 * DiscardRaceRunner, so the timeout (T4), the single-flight flag (T6/A26)
 * and the fallback semantics (T3/A7/A8) apply to the full image, not per tile.
 *
 * All timing is via FakeTimeSource (rule X5): the timer only fires when the
 * test releases it, and the slow adapter blocks on latches — no real sleeps.
 */
public class DenoiseModuleTiledTest {

    private static final int W = 300;
    private static final int H = 280;
    private static final int C = 3;
    private static final long TIMEOUT_MS = 200L;

    /**
     * Image larger than one tile: 300x280 with tileSize=256, halo=17, blend=16
     * gives stride 206 and exactly 4 tiles (2x2). Asserted via adapter callCount.
     */
    private static final int EXPECTED_TILES = 4;

    /**
     * Fake LiteRT tile adapter. Output is always input + 1 per element, so the
     * tests can tell "pixels came from the model" (in+1) from a fusion-only
     * fallback (pixels unchanged). BLOCK parks the first tile on a latch so the
     * whole tiler.process call is late; THROW makes inference fail.
     */
    private static class FakeTileAdapter implements LiteRtAdapter {
        enum Behavior { PLUS_ONE, BLOCK, THROW }

        final CountDownLatch entered = new CountDownLatch(1);
        final CountDownLatch proceed = new CountDownLatch(1);
        final AtomicInteger callCount = new AtomicInteger(0);
        volatile Behavior behavior = Behavior.PLUS_ONE;

        @Override
        public void run(ByteBuffer in, ByteBuffer out) {
            callCount.incrementAndGet();
            entered.countDown();
            Behavior b = behavior;
            if (b == Behavior.THROW) {
                throw new RuntimeException("tile inference exploded");
            }
            if (b == Behavior.BLOCK) {
                try {
                    proceed.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(e);
                }
            }
            FloatBuffer inF = in.asFloatBuffer();
            inF.rewind();
            FloatBuffer outF = out.asFloatBuffer();
            outF.clear();
            while (inF.hasRemaining()) {
                outF.put(inF.get() + 1f);
            }
        }

        @Override
        public void close() {
        }
    }

    private static class RecordingErrorListener implements ErrorListener {
        final AtomicReference<String> lastError = new AtomicReference<>();
        @Override
        public void onError(String msg, Throwable t) {
            lastError.set(msg);
        }
    }

    private static float[] rampImage() {
        float[] image = new float[W * H * C];
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                image[(y * W + x) * C + 0] = x / (float) W;
                image[(y * W + x) * C + 1] = y / (float) H;
                image[(y * W + x) * C + 2] = 0.5f;
            }
        }
        return image;
    }

    private static FusedFrame frame(String burstId, float[] image) {
        return new FusedFrame(burstId, "naive", image, "fusion_multi");
    }

    private static DenoiseModule module(FakeTileAdapter adapter, FakeTimeSource time,
                                        RecordingErrorListener listener) {
        return new DenoiseModule(
            f -> new float[0], // non-tiled runner stub; never used by the tiled tests
            time,
            DenoiseModule.MODEL_VARIANT_CNN_V1,
            DenoiseModule.PRECISION_FP32,
            listener,
            new Tiler(),
            adapter);
    }

    @Test
    public void tiledImageLargerThanOneTileFinishesInTime() {
        FakeTileAdapter adapter = new FakeTileAdapter();
        FakeTimeSource time = new FakeTimeSource();
        RecordingErrorListener listener = new RecordingErrorListener();
        DenoiseModule module = module(adapter, time, listener);

        float[] image = rampImage();
        RaceResult result = module.denoise(frame("burst-tiled-ok", image),
            new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE);

        assertTrue("Fast tiled inference must win the race", result.isInferenceWon());
        assertFalse(result.isTimerWon());
        assertEquals("No fallback reason when inference wins", null, result.getFallbackReason());

        DenoisedFrame out = result.getFrame();
        assertFalse("Timer did not win, no timeout flag", out.isTimeoutOccurred());
        assertFalse("No discard race when inference wins", out.isDiscardRaceOccurred());
        assertEquals(W * H * C, out.getImage().length);
        assertEquals("burst-tiled-ok", out.getBurstId());
        assertTrue("Latency must be non-negative", out.getLatencyNs() >= 0);

        assertEquals("Every tile must go through the adapter", EXPECTED_TILES, adapter.callCount.get());

        // Every delivered pixel must be model output (input+1), not a fallback copy.
        float maxDiff = 0f;
        for (int i = 0; i < image.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs((image[i] + 1f) - out.getImage()[i]));
        }
        assertTrue("Output must equal model output (input+1), max diff " + maxDiff,
            maxDiff <= 1e-4f);
    }

    @Test
    public void tiledTooSlowTimesOutFusionOnlyAndLateResultDiscarded() throws Exception {
        FakeTileAdapter adapter = new FakeTileAdapter();
        adapter.behavior = FakeTileAdapter.Behavior.BLOCK;
        FakeTimeSource time = new FakeTimeSource();
        RecordingErrorListener listener = new RecordingErrorListener();
        DenoiseModule module = module(adapter, time, listener);

        float[] image = rampImage();
        AtomicReference<RaceResult> firstRef = new AtomicReference<>();
        CountDownLatch firstDone = new CountDownLatch(1);
        new Thread(() -> {
            firstRef.set(module.denoise(frame("burst-slow", image),
                new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE));
            firstDone.countDown();
        }, "tiled-first-burst").start();

        assertTrue("Tiled inference must have started", adapter.entered.await(5, TimeUnit.SECONDS));
        time.releaseSleep(); // timer fires -> timeout
        assertTrue("denoise must return when the timer wins", firstDone.await(5, TimeUnit.SECONDS));

        RaceResult result = firstRef.get();
        assertTrue(result.isTimerWon());
        assertFalse(result.isInferenceWon());

        DenoisedFrame out = result.getFrame();
        assertTrue("Timer win must set the timeout flag (T3)", out.isTimeoutOccurred());
        assertFalse("At delivery the late result has not been discarded yet", out.isDiscardRaceOccurred());
        assertEquals(RaceResult.REASON_TIMEOUT, result.getFallbackReason());
        assertEquals("latencyNs on a timeout is the time-to-deliver (fake clock = timeout)",
            TIMEOUT_MS * 1_000_000L, out.getLatencyNs());
        assertTrue("Fallback frame must carry the fusion-only pixels unchanged",
            Arrays.equals(image, out.getImage()));

        // Now let the late tiled inference finish: its result must be discarded, never delivered.
        adapter.proceed.countDown();
        assertTrue("Late inference must complete and be marked discarded",
            result.awaitLateInference(5000));
        assertTrue("Timer won AND late result dropped => discardRaceEvent (T3/A7)",
            result.isDiscardRaceOccurred());

        // Single-flight released from the late-inference thread: next burst runs normally (X4/T6).
        RaceResult next = module.denoise(frame("burst-after", rampImage()),
            new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE);
        assertTrue("Next burst after timeout + late completion must run normally",
            next.isInferenceWon());
    }

    @Test
    public void tiledSecondBurstWhileLateTiledInferenceStillRunningGetsBusy() throws Exception {
        FakeTileAdapter adapter = new FakeTileAdapter();
        adapter.behavior = FakeTileAdapter.Behavior.BLOCK;
        FakeTimeSource time = new FakeTimeSource();
        RecordingErrorListener listener = new RecordingErrorListener();
        DenoiseModule module = module(adapter, time, listener);

        float[] image = rampImage();
        AtomicReference<RaceResult> firstRef = new AtomicReference<>();
        CountDownLatch firstDone = new CountDownLatch(1);
        new Thread(() -> {
            firstRef.set(module.denoise(frame("burst-busy-1", image),
                new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE));
            firstDone.countDown();
        }, "tiled-first-burst").start();

        assertTrue(adapter.entered.await(5, TimeUnit.SECONDS));
        time.releaseSleep();
        assertTrue(firstDone.await(5, TimeUnit.SECONDS));
        assertTrue(firstRef.get().isTimerWon());

        // Late tiled inference is still parked on the first tile: the flag is held.
        float[] image2 = new float[W * H * C];
        Arrays.fill(image2, 0.25f);
        RaceResult second = module.denoise(frame("burst-busy-2", image2),
            new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE);

        assertEquals("Second burst while busy must be a BUSY fallback (T6/A26)",
            RaceResult.REASON_BUSY, second.getFallbackReason());
        DenoisedFrame secondFrame = second.getFrame();
        assertTrue("BUSY fallback is fusion-only (A8)", secondFrame.isTimeoutOccurred());
        assertFalse(secondFrame.isDiscardRaceOccurred());
        assertEquals("BUSY fallback latency is 0 (no inference measured)", 0L,
            secondFrame.getLatencyNs());
        assertTrue("BUSY fallback must carry its own frame's pixels unchanged",
            Arrays.equals(image2, secondFrame.getImage()));

        // Cleanup: finish the late inference.
        adapter.proceed.countDown();
        assertTrue(firstRef.get().awaitLateInference(5000));
    }

    @Test
    public void tiledInferenceExceptionDeliversFusionOnlyWithInferenceError() {
        FakeTileAdapter adapter = new FakeTileAdapter();
        adapter.behavior = FakeTileAdapter.Behavior.THROW;
        FakeTimeSource time = new FakeTimeSource();
        RecordingErrorListener listener = new RecordingErrorListener();
        DenoiseModule module = module(adapter, time, listener);

        float[] image = rampImage();
        RaceResult result = module.denoise(frame("burst-err", image),
            new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE);

        assertEquals("Inference failure must surface as INFERENCE_ERROR fallback",
            RaceResult.REASON_INFERENCE_ERROR, result.getFallbackReason());
        assertFalse(result.isInferenceWon());
        assertFalse(result.isTimerWon());

        DenoisedFrame out = result.getFrame();
        assertTrue("Fallback frame marks fusion-only pixels (A8)", out.isTimeoutOccurred());
        assertFalse(out.isDiscardRaceOccurred());
        assertTrue("Fallback frame must carry the fusion-only pixels unchanged",
            Arrays.equals(image, out.getImage()));
        assertNotNull("The error must reach the ErrorListener", listener.lastError.get());
    }
}
