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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.*;

/**
 * Whole-image tiled path under the discard-race (Tasks 1 + 5).
 *
 * The tiled overload runs tiler.process as the InferenceRunner inside
 * DiscardRaceRunner, so the timeout (T4), the single-flight flag (T6/A26)
 * and the fallback semantics (T3/A7/A8) apply to the full image, not per tile.
 * T9: after the timer wins, the late tiled run abandons cooperatively at the
 * next tile boundary (never interrupting the tile in flight, T1).
 *
 * All timing is via FakeTimeSource (rule X5): the timer only fires when the
 * test releases it, and slow tiles block on per-call latches — no real sleeps.
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
     * fallback (pixels unchanged).
     *
     * callGates.get(i) gates the i-th run() call (missing entry or null passes
     * through) — this models per-tile latency precisely, which the abandon
     * tests need: tile 1 completes, tile 2 would hang forever without T9.
     * throwAfterGate makes a call throw after its gate (models the in-flight
     * tile failing; a tile in flight cannot be abandoned, rule T1).
     */
    private static class FakeTileAdapter implements LiteRtAdapter {
        volatile List<CountDownLatch> callGates = new ArrayList<>();
        volatile boolean throwAfterGate = false;
        final AtomicInteger callCount = new AtomicInteger(0);
        final CountDownLatch firstCallEntered = new CountDownLatch(1);

        @Override
        public void run(ByteBuffer in, ByteBuffer out) {
            int callIndex = callCount.getAndIncrement();
            if (callIndex == 0) {
                firstCallEntered.countDown();
            }
            List<CountDownLatch> gates = callGates;
            CountDownLatch gate =
                gates != null && callIndex < gates.size() ? gates.get(callIndex) : null;
            if (gate != null) {
                try {
                    gate.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(e);
                }
            }
            if (throwAfterGate) {
                throw new RuntimeException("tile inference exploded");
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

    private static float maxAbsDiff(float[] a, float[] b) {
        float maxDiff = 0f;
        for (int i = 0; i < a.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
        }
        return maxDiff;
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
        assertEquals("Output must equal model output (input+1)",
            0f, maxAbsDiff(image, out.getImage()) - 1f, 1e-4f);
    }

    @Test
    public void tiledTooSlowTimesOutFusionOnlyAndLateResultDiscarded() throws Exception {
        FakeTileAdapter adapter = new FakeTileAdapter();
        CountDownLatch tile1Gate = new CountDownLatch(1);
        adapter.callGates = Arrays.asList(tile1Gate);
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

        assertTrue("Tiled inference must have started", adapter.firstCallEntered.await(5, TimeUnit.SECONDS));
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

        // Let the in-flight tile finish: the late run abandons at the tile boundary (T9)
        // and its (never-completed) result is discarded, never delivered.
        tile1Gate.countDown();
        assertTrue("Late run must end (abandoned) and be marked discarded",
            result.awaitLateInference(5000));
        assertTrue("Timer won AND late result dropped => discardRaceEvent (T3/T9)",
            result.isDiscardRaceOccurred());

        // Single-flight released from the late-inference thread: next burst runs normally (X4/T6).
        adapter.callGates = new ArrayList<>();
        float[] image2 = rampImage();
        RaceResult next = module.denoise(frame("burst-after", image2),
            new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE);
        assertTrue("Next burst after timeout + late completion must run normally",
            next.isInferenceWon());
        assertEquals("Next burst output must be model output (input+1)",
            0f, maxAbsDiff(image2, next.getFrame().getImage()) - 1f, 1e-4f);
    }

    @Test
    public void tiledSecondBurstWhileLateTiledInferenceStillRunningGetsBusy() throws Exception {
        FakeTileAdapter adapter = new FakeTileAdapter();
        CountDownLatch tile1Gate = new CountDownLatch(1);
        adapter.callGates = Arrays.asList(tile1Gate);
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

        assertTrue(adapter.firstCallEntered.await(5, TimeUnit.SECONDS));
        time.releaseSleep();
        assertTrue(firstDone.await(5, TimeUnit.SECONDS));
        assertTrue(firstRef.get().isTimerWon());

        // Late tiled inference is still parked inside tile 1: the flag is held.
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

        // Cleanup: finish the in-flight tile; the late run abandons (T9).
        tile1Gate.countDown();
        assertTrue(firstRef.get().awaitLateInference(5000));
    }

    @Test
    public void tiledInferenceExceptionDeliversFusionOnlyWithInferenceError() {
        FakeTileAdapter adapter = new FakeTileAdapter();
        adapter.throwAfterGate = true;
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

    // ---- T9: cooperative abandon between tiles ----

    @Test
    public void tiledTimeoutAbandonsRemainingTilesAtTileBoundaryAndNextBurstRuns() throws Exception {
        FakeTileAdapter adapter = new FakeTileAdapter();
        CountDownLatch tile1Gate = new CountDownLatch(1);
        adapter.callGates = Arrays.asList(tile1Gate);
        FakeTimeSource time = new FakeTimeSource();
        RecordingErrorListener listener = new RecordingErrorListener();
        DenoiseModule module = module(adapter, time, listener);

        float[] image = rampImage();
        AtomicReference<RaceResult> firstRef = new AtomicReference<>();
        CountDownLatch firstDone = new CountDownLatch(1);
        new Thread(() -> {
            firstRef.set(module.denoise(frame("burst-abandon", image),
                new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE));
            firstDone.countDown();
        }, "tiled-first-burst").start();

        assertTrue(adapter.firstCallEntered.await(5, TimeUnit.SECONDS));
        time.releaseSleep(); // timer wins; abandon flag set
        assertTrue(firstDone.await(5, TimeUnit.SECONDS));
        assertTrue(firstRef.get().isTimerWon());

        // Finish the in-flight tile: the late run must stop BEFORE tile 2 (T9),
        // so exactly 1 adapter call happened for this burst — never 4.
        tile1Gate.countDown();
        assertTrue("Late run must end (abandoned)", firstRef.get().awaitLateInference(5000));
        assertEquals("Remaining tiles must NOT be processed after the timer won (T9)",
            1, adapter.callCount.get());
        assertTrue("Abandoned late run => discardRaceOccurred (T3/T9)",
            firstRef.get().isDiscardRaceOccurred());

        // Busy flag was released by the (abandoned) late run: next burst runs normally.
        adapter.callGates = new ArrayList<>();
        float[] image2 = rampImage();
        RaceResult next = module.denoise(frame("burst-after-abandon", image2),
            new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE);
        assertTrue("Next burst after an abandoned late run must run normally",
            next.isInferenceWon());
        assertEquals("Next burst output must be model output (input+1)",
            0f, maxAbsDiff(image2, next.getFrame().getImage()) - 1f, 1e-4f);
    }

    @Test
    public void tiledBenchmarkWaitReturnsQuicklyBecauseLateRunAbandons() throws Exception {
        FakeTileAdapter adapter = new FakeTileAdapter();
        CountDownLatch tile1Gate = new CountDownLatch(1);
        CountDownLatch tile2Gate = new CountDownLatch(1); // never released: without T9 the
        adapter.callGates = Arrays.asList(tile1Gate, tile2Gate); // late run would hang here
        FakeTimeSource time = new FakeTimeSource();
        RecordingErrorListener listener = new RecordingErrorListener();
        DenoiseModule module = module(adapter, time, listener);

        float[] image = rampImage();
        AtomicReference<RaceResult> firstRef = new AtomicReference<>();
        CountDownLatch firstDone = new CountDownLatch(1);
        new Thread(() -> {
            firstRef.set(module.denoise(frame("burst-bench-1", image),
                new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.BENCHMARK));
            firstDone.countDown();
        }, "tiled-first-burst").start();

        assertTrue(adapter.firstCallEntered.await(5, TimeUnit.SECONDS));
        time.releaseSleep(); // timer wins
        assertTrue(firstDone.await(5, TimeUnit.SECONDS));
        assertTrue(firstRef.get().isTimerWon());

        // Finish the in-flight tile; without T9 the late run would park on tile2Gate
        // forever and the BENCHMARK-mode bounded wait below would stall on it.
        tile1Gate.countDown();
        assertTrue(firstRef.get().awaitLateInference(5000));
        assertEquals("Late run abandoned after the in-flight tile (T9)", 1, adapter.callCount.get());

        // The BENCHMARK wait for the previous burst must return immediately now.
        adapter.callGates = new ArrayList<>();
        float[] image2 = rampImage();
        AtomicReference<RaceResult> secondRef = new AtomicReference<>();
        CountDownLatch secondDone = new CountDownLatch(1);
        new Thread(() -> {
            secondRef.set(module.denoise(frame("burst-bench-2", image2),
                new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.BENCHMARK));
            secondDone.countDown();
        }, "tiled-second-burst").start();
        assertTrue("BENCHMARK wait must return quickly once the late run has ended (T9/T6)",
            secondDone.await(5, TimeUnit.SECONDS));
        assertTrue("Second BENCHMARK burst must run normally", secondRef.get().isInferenceWon());
    }

    @Test
    public void tiledLateRunThrowsAfterTimerWonBusyStillReleased() throws Exception {
        FakeTileAdapter adapter = new FakeTileAdapter();
        CountDownLatch tile1Gate = new CountDownLatch(1);
        adapter.callGates = Arrays.asList(tile1Gate);
        adapter.throwAfterGate = true; // the in-flight tile fails after the timer already won
        FakeTimeSource time = new FakeTimeSource();
        RecordingErrorListener listener = new RecordingErrorListener();
        DenoiseModule module = module(adapter, time, listener);

        float[] image = rampImage();
        AtomicReference<RaceResult> firstRef = new AtomicReference<>();
        CountDownLatch firstDone = new CountDownLatch(1);
        new Thread(() -> {
            firstRef.set(module.denoise(frame("burst-late-throw", image),
                new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE));
            firstDone.countDown();
        }, "tiled-first-burst").start();

        assertTrue(adapter.firstCallEntered.await(5, TimeUnit.SECONDS));
        time.releaseSleep(); // timer wins while tile 1 is in flight
        assertTrue(firstDone.await(5, TimeUnit.SECONDS));
        assertTrue(firstRef.get().isTimerWon());

        // The in-flight tile cannot be abandoned (T1); it finishes its gate and throws.
        // The late run ends exceptionally; its (nonexistent) result is dropped, the
        // busy flag is still released, and the next burst must run normally.
        tile1Gate.countDown();
        firstRef.get().awaitLateInference(5000);
        assertEquals("Late run ended on the in-flight tile (throw); abandon preempts tile 2",
            1, adapter.callCount.get());

        adapter.callGates = new ArrayList<>();
        adapter.throwAfterGate = false;
        float[] image2 = rampImage();
        RaceResult next = module.denoise(frame("burst-after-throw", image2),
            new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE);
        assertTrue("Busy flag must be released even when the late run throws (finally-release)",
            next.isInferenceWon());
        assertEquals("Next burst output must be model output (input+1)",
            0f, maxAbsDiff(image2, next.getFrame().getImage()) - 1f, 1e-4f);
    }
}
