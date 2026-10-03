package com.prelude.denoise;

import com.prelude.denoise.model.DenoisedFrame;
import com.prelude.denoise.model.FusedFrame;
import com.prelude.denoise.model.FrameGeometry;
import com.prelude.denoise.model.LiteRtAdapter;
import com.prelude.denoise.model.ModelSource;
import com.prelude.denoise.timeout.FakeTimeSource;
import com.prelude.denoise.timeout.RaceResult;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * JVM tests for DenoiseService (F2): fake ModelSource + fake tile adapter,
 * fake time source, no real sleeps (X5).
 */
public class DenoiseServiceTest {

    private static final int W = 300, H = 280, C = 3;
    private static final long TIMEOUT_MS = 200L;

    /** Fake ModelSource: hands out a small direct buffer, counts loads. */
    static class FakeModelSource implements ModelSource {
        final AtomicInteger loadCount = new AtomicInteger(0);

        @Override
        public ByteBuffer loadModel() {
            loadCount.incrementAndGet();
            return ByteBuffer.allocateDirect(64).order(ByteOrder.nativeOrder());
        }
    }

    /** Fake tile adapter: per-call gates, +1 output, close flag (mirrors TiledTest's). */
    static class FakeTileAdapter implements LiteRtAdapter {
        volatile List<CountDownLatch> callGates = new ArrayList<>();
        final AtomicInteger callCount = new AtomicInteger(0);
        final CountDownLatch firstCallEntered = new CountDownLatch(1);
        volatile boolean closed = false;

        @Override
        public void run(ByteBuffer in, ByteBuffer out) {
            int idx = callCount.getAndIncrement();
            if (idx == 0) {
                firstCallEntered.countDown();
            }
            List<CountDownLatch> gates = callGates;
            CountDownLatch gate = gates != null && idx < gates.size() ? gates.get(idx) : null;
            if (gate != null) {
                try {
                    gate.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(e);
                }
            }
            java.nio.FloatBuffer inF = in.asFloatBuffer();
            inF.rewind();
            java.nio.FloatBuffer outF = out.asFloatBuffer();
            outF.clear();
            while (inF.hasRemaining()) {
                outF.put(inF.get() + 1f);
            }
        }

        @Override
        public void close() {
            closed = true;
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

    private static float maxAbsDiff(float[] a, float[] b) {
        float maxDiff = 0f;
        for (int i = 0; i < a.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
        }
        return maxDiff;
    }

    @Test
    public void endToEnd300x280IdentityModel() throws Exception {
        FakeModelSource source = new FakeModelSource();
        FakeTileAdapter adapter = new FakeTileAdapter();
        FakeTimeSource time = new FakeTimeSource();
        DenoiseService service = new DenoiseService(source, adapter, time, null);

        float[] image = rampImage();
        RaceResult result = service.denoise(frame("svc-e2e", image),
            new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE);

        assertTrue("Identity tiled inference must win", result.isInferenceWon());
        DenoisedFrame out = result.getFrame();
        assertFalse(out.isTimeoutOccurred());
        assertFalse(out.isDiscardRaceOccurred());
        assertEquals("svc-e2e", out.getBurstId());
        assertEquals(4, adapter.callCount.get()); // 300x280 = 4 tiles at stride 206
        assertEquals("Output must be model output (input+1)",
            0f, maxAbsDiff(image, out.getImage()) - 1f, 1e-4f);
        assertEquals("Model source must be read exactly once", 1, source.loadCount.get());
    }

    @Test
    public void timeoutPathDeliversFusionOnlyAndDiscardsLateTiledResult() throws Exception {
        FakeTileAdapter adapter = new FakeTileAdapter();
        CountDownLatch tile1Gate = new CountDownLatch(1);
        adapter.callGates = Arrays.asList(tile1Gate);
        FakeTimeSource time = new FakeTimeSource();
        DenoiseService service = new DenoiseService(new FakeModelSource(), adapter, time, null);

        float[] image = rampImage();
        AtomicReference<RaceResult> firstRef = new AtomicReference<>();
        CountDownLatch firstDone = new CountDownLatch(1);
        new Thread(() -> {
            firstRef.set(service.denoise(frame("svc-timeout", image),
                new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE));
            firstDone.countDown();
        }).start();

        assertTrue(adapter.firstCallEntered.await(5, TimeUnit.SECONDS));
        time.releaseSleep();
        assertTrue(firstDone.await(5, TimeUnit.SECONDS));

        RaceResult result = firstRef.get();
        assertTrue(result.isTimerWon());
        assertTrue("Timer win must set the timeout flag (T3)", result.getFrame().isTimeoutOccurred());
        assertEquals(TIMEOUT_MS * 1_000_000L, result.getFrame().getLatencyNs());
        assertArrayEquals("Fallback frame carries fusion-only pixels",
            image, result.getFrame().getImage(), 1e-9f);

        // In-flight tile finishes -> late run abandons at the tile boundary (T9).
        tile1Gate.countDown();
        assertTrue(result.awaitLateInference(5000));
        assertTrue(result.isDiscardRaceOccurred());
        assertEquals(1, adapter.callCount.get());

        // Next burst runs normally.
        adapter.callGates = new ArrayList<>();
        float[] image2 = rampImage();
        RaceResult next = service.denoise(frame("svc-after", image2),
            new FrameGeometry(W, H, C), TIMEOUT_MS, DenoiseModule.RunMode.LIVE);
        assertTrue(next.isInferenceWon());
    }

    @Test
    public void uncalibratedConvenienceUsesCeilingAndLogsLoudly() {
        FakeTileAdapter adapter = new FakeTileAdapter();
        FakeTimeSource time = new FakeTimeSource();
        List<String> infos = new ArrayList<>();
        DenoiseService service = new DenoiseService(new FakeModelSource(), adapter, time,
            new com.prelude.denoise.timeout.ErrorListener() {
                @Override
                public void onError(String msg, Throwable t) {
                    infos.add(msg);
                }
                @Override
                public void onInfo(String msg) {
                    infos.add(msg);
                }
            });

        RaceResult result = service.denoise(frame("svc-uncal", rampImage()),
            new FrameGeometry(W, H, C), DenoiseModule.RunMode.LIVE);

        assertTrue(result.isInferenceWon());
        assertTrue("The uncalibrated ceiling must be loud-logged (rule T4)",
            infos.stream().anyMatch(m -> m.contains("UNCALIBRATED")));
    }

    @Test
    public void closeReleasesTheAdapter() {
        FakeTileAdapter adapter = new FakeTileAdapter();
        DenoiseService service = new DenoiseService(new FakeModelSource(), adapter,
            new FakeTimeSource(), null);

        assertFalse(adapter.closed);
        service.close();
        assertTrue("close() must close the underlying adapter", adapter.closed);
    }
}
