package com.prelude.denoise.model;

import com.prelude.denoise.DenoiseModule;
import com.prelude.denoise.timeout.ErrorListener;
import com.prelude.denoise.timeout.RaceResult;
import com.prelude.denoise.timeout.SystemTimeSource;
import com.prelude.denoise.tiling.Tiler;
import com.prelude.denoise.tiling.TileInferenceAdapter;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.*;

public class LiteRtInferenceRunnerTest {

    /**
     * Fake adapter with latch-based synchronization.
     * enteredLatch counts down when run() has started (signals the test).
     * runLatch blocks run() until the test releases it.
     */
    private static class FakeAdapter implements LiteRtAdapter {
        final CountDownLatch enteredLatch;
        final CountDownLatch runLatch;
        volatile boolean closed = false;

        FakeAdapter(CountDownLatch enteredLatch, CountDownLatch runLatch) {
            this.enteredLatch = enteredLatch;
            this.runLatch = runLatch;
        }

        @Override
        public void run(ByteBuffer in, ByteBuffer out) {
            assertTrue("Must use direct buffer", in.isDirect());
            assertTrue("Must use direct buffer", out.isDirect());
            enteredLatch.countDown(); // signal: run has started
            try {
                runLatch.await(); // block until test releases
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            out.clear();
            in.clear();
            out.put(in);
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class FakeErrorListener implements ErrorListener {
        volatile String lastError = null;
        volatile String lastInfo = null;
        @Override
        public void onError(String msg, Throwable e) {
            lastError = msg;
        }
        @Override
        public void onInfo(String msg) {
            lastInfo = msg;
        }
    }

    @Test
    public void testCloseWaitsForLateInference() throws InterruptedException {
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch block = new CountDownLatch(1);
        FakeAdapter adapter = new FakeAdapter(entered, block);
        FakeErrorListener errListener = new FakeErrorListener();
        LiteRtInferenceRunner runner = new LiteRtInferenceRunner(adapter, errListener, 2, 2, 3);

        Thread t = new Thread(() ->
            runner.run(new FusedFrame("burst", "s1", new float[12], "fusion_single"))
        );
        t.start();

        entered.await(); // wait until run() has started and holds the lock

        CountDownLatch closeDone = new CountDownLatch(1);
        Thread closer = new Thread(() -> {
            runner.close();
            closeDone.countDown();
        });
        closer.start();

        // close() should be blocked waiting for the semaphore
        assertFalse("Close must not finish while run is active",
            closeDone.await(100, java.util.concurrent.TimeUnit.MILLISECONDS));
        assertFalse("Adapter must not be closed yet", adapter.closed);

        block.countDown(); // unblock run()
        closeDone.await(2000, java.util.concurrent.TimeUnit.MILLISECONDS);

        assertTrue("Close should have finished", adapter.closed);
        assertNull(errListener.lastError);

        t.join(1000);
    }

    @Test
    public void testCloseTimesOutAndLeaks() throws InterruptedException {
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch block = new CountDownLatch(1); // never released during close
        FakeAdapter adapter = new FakeAdapter(entered, block);
        FakeErrorListener errListener = new FakeErrorListener();
        LiteRtInferenceRunner runner = new LiteRtInferenceRunner(adapter, errListener, 2, 2, 3);

        Thread t = new Thread(() ->
            runner.run(new FusedFrame("burst", "s1", new float[12], "fusion_single"))
        );
        t.start();

        try {
            entered.await(); // wait until run() has started and holds the lock
            runner.close();  // waits 500ms, times out

            assertFalse("Should leak instead of closing", adapter.closed);
            assertNotNull(errListener.lastError);
            assertTrue(errListener.lastError.contains("leaking"));
        } finally {
            block.countDown(); // always unblock the thread so it can exit
            t.join(2000);
        }
    }

    @Test(expected = IllegalStateException.class)
    public void testRejectsConcurrentRuns() throws InterruptedException {
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch block = new CountDownLatch(1);
        FakeAdapter adapter = new FakeAdapter(entered, block);
        LiteRtInferenceRunner runner = new LiteRtInferenceRunner(adapter, null, 2, 2, 3);

        Thread t = new Thread(() ->
            runner.run(new FusedFrame("burst", "s1", new float[12], "fusion_single"))
        );
        t.start();

        try {
            entered.await(); // wait until first run() holds the lock
            runner.run(new FusedFrame("burst", "s1", new float[12], "fusion_single"));
        } finally {
            block.countDown();
            t.join(1000);
        }
    }

    @Test
    public void testEndToEndCallPathWithFakeNCHW() {
        int w = 2, h = 2, c = 3;
        float[] hwcInput = new float[12];
        hwcInput[3] = 42f; // c=0 of (x=1,y=0)

        LiteRtAdapter adapter = new LiteRtAdapter() {
            @Override
            public void run(ByteBuffer in, ByteBuffer out) {
                FloatBuffer inF = in.asFloatBuffer();
                // NCHW: ch=0, y=0, x=1 -> index 1
                assertEquals(42f, inF.get(1), 1e-6f);
                out.asFloatBuffer().put(inF);
            }
            @Override
            public void close() {}
        };

        LiteRtInferenceRunner runner = new LiteRtInferenceRunner(adapter, null, w, h, c);
        float[] hwcOutput = runner.run(new FusedFrame("burst", "s1", hwcInput, "fusion_single"));
        assertEquals(42f, hwcOutput[3], 1e-6f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRejectsNonDirectByteBuffer() {
        ByteBuffer nonDirect = ByteBuffer.allocate(64);
        new LiteRtAdapterImpl(nonDirect, null);
    }

    /**
     * Task 2 companion property: the runner reuses its internal scratch arrays,
     * but the RETURNED array must still be fresh per call — it escapes into the
     * delivered DenoisedFrame, so a later run must never alias or overwrite it.
     */
    @Test
    public void testRunReturnsAFreshArrayEachCallAndNeverAliasesPreviousOutput() {
        int w = 2, h = 2, c = 1;
        LiteRtAdapter echo = new LiteRtAdapter() {
            @Override
            public void run(ByteBuffer in, ByteBuffer out) {
                in.rewind();
                out.clear();
                out.put(in);
            }
            @Override
            public void close() {
            }
        };
        LiteRtInferenceRunner runner = new LiteRtInferenceRunner(echo, null, w, h, c);

        float[] in1 = {1f, 2f, 3f, 4f};
        float[] in2 = {5f, 6f, 7f, 8f};
        float[] out1 = runner.run(new FusedFrame("b", "s1", in1, "fusion_multi"));
        float[] out2 = runner.run(new FusedFrame("b", "s1", in2, "fusion_multi"));

        assertNotSame("Returned array must be fresh per call (it escapes into DenoisedFrame)",
            out1, out2);
        assertArrayEquals("Earlier output must be untouched by the second run", in1, out1, 1e-6f);
        assertArrayEquals(in2, out2, 1e-6f);
    }

    /**
     * Full path: FusedFrame → Tiler → HWC→NCHW → adapter → NCHW→HWC → DenoisedFrame.
     * Uses a 300x280 image (larger than one tile at tileSize=256).
     * The fake adapter passes through the data unchanged (identity model).
     * Tiled output must match a full-frame reference within float precision.
     */
    @Test
    public void testTiledCallPathLargerThanOneTile() {
        int w = 300, h = 280, c = 3;
        float[] image = new float[w * h * c];
        // Asymmetric pattern: ramp across width in channel 0
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                image[(y * w + x) * c + 0] = x / (float) w;      // R ramp
                image[(y * w + x) * c + 1] = y / (float) h;      // G ramp
                image[(y * w + x) * c + 2] = 0.5f;                // B constant
            }
        }

        // Identity adapter: NCHW pass-through (copy in→out)
        LiteRtAdapter passthrough = new LiteRtAdapter() {
            @Override
            public void run(ByteBuffer in, ByteBuffer out) {
                assertTrue("Must use direct buffer", in.isDirect());
                in.rewind();
                out.clear();
                out.put(in);
            }
            @Override
            public void close() {}
        };

        Tiler tiler = new Tiler(); // 256, halo=17, blend=16
        FrameGeometry geom = new FrameGeometry(w, h, c);

        // Use DenoiseModule.denoise(FusedFrame, FrameGeometry, ...) path
        FusedFrame input = new FusedFrame("burst-tiled", "s1", image, "fusion_single");
        DenoiseModule module = new DenoiseModule(
            f -> new float[0], // unused InferenceRunner stub
            SystemTimeSource.INSTANCE,
            DenoiseModule.MODEL_VARIANT_CNN_V1,
            DenoiseModule.PRECISION_FP32,
            null,
            tiler,
            passthrough);

        RaceResult result = module.denoise(input, geom, 500, DenoiseModule.RunMode.LIVE);
        DenoisedFrame frame = result.getFrame();

        // Check non-trivial output
        float[] out = frame.getImage();
        assertEquals(w * h * c, out.length);
        assertFalse("Output must not be all zeros", allEqual(out, 0f));
        assertFalse("Output must not be all same", allEqual(out, out[0]));

        // Compare against full-frame reference (identity: tiled pass-through of HWC→NCHW→HWC
        // should reconstruct the input within tiling tolerance)
        float maxDiff = 0f;
        for (int i = 0; i < image.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(image[i] - out[i]));
        }
        // Tiling with blending introduces small numerical differences
        assertTrue("Max diff " + maxDiff + " exceeds tolerance", maxDiff < 0.01f);
        assertTrue("Latency must be > 0", frame.getLatencyNs() > 0);
        assertEquals("burst-tiled", frame.getBurstId());
        assertTrue("Inference must win when the model is fast", result.isInferenceWon());
    }

    /**
     * Mutation check: halo 0 in the tiled path must produce different output.
     * Uses a spatial 3x3 convolution adapter so halo guard-band effects
     * (receptive field padding artifacts at boundaries) become visible.
     */
    @Test
    public void testTiledPathHaloZeroMismatch() {
        int w = 300, h = 280, c = 3;
        float[] image = new float[w * h * c];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                image[(y * w + x) * c] = (x + y) * 0.01f;
                image[(y * w + x) * c + 1] = (x - y + 300) * 0.005f;
                image[(y * w + x) * c + 2] = 0.3f;
            }
        }

        // Spatial 3x3 conv: sensitive to boundary padding (receptive field > 1)
        LiteRtAdapter convAdapter = new LiteRtAdapter() {
            @Override
            public void run(ByteBuffer in, ByteBuffer out) {
                FloatBuffer inF = in.asFloatBuffer();
                FloatBuffer outF = out.asFloatBuffer();
                inF.rewind(); outF.clear();
                int ts = 256;
                float[] tileChw = new float[c * ts * ts];
                inF.get(tileChw);
                float[] outChw = new float[c * ts * ts];
                for (int ch = 0; ch < c; ch++) {
                    int offset = ch * ts * ts;
                    for (int y = 0; y < ts; y++) {
                        for (int x = 0; x < ts; x++) {
                            float sum = 0f;
                            for (int dy = -1; dy <= 1; dy++) {
                                int ny = y + dy;
                                for (int dx = -1; dx <= 1; dx++) {
                                    int nx = x + dx;
                                    if (ny >= 0 && ny < ts && nx >= 0 && nx < ts) {
                                        sum += tileChw[offset + ny * ts + nx];
                                    }
                                }
                            }
                            outChw[offset + y * ts + x] = sum / 9.0f;
                        }
                    }
                }
                outF.put(outChw);
            }
            @Override
            public void close() {}
        };

        Tiler goodTiler = new Tiler(256, 17, 16);
        Tiler badTiler  = new Tiler(256, 0, 16); // halo 0

        TileInferenceAdapter goodAdapter = new TileInferenceAdapter(convAdapter, 256, c);
        TileInferenceAdapter badAdapter  = new TileInferenceAdapter(convAdapter, 256, c);

        float[] goodOut = goodTiler.process(image, w, h, c, goodAdapter);
        float[] badOut  = badTiler.process(image, w, h, c, badAdapter);

        // With halo=0, boundary pixels computed with edge padding are blended in,
        // producing different results than halo=17 where boundary artifacts are dropped.
        boolean differ = false;
        for (int i = 0; i < goodOut.length; i++) {
            if (Math.abs(goodOut[i] - badOut[i]) > 1e-4f) {
                differ = true;
                break;
            }
        }
        assertTrue("Halo 0 must produce different output from halo 17", differ);
    }

    private static boolean allEqual(float[] arr, float val) {
        for (float v : arr) {
            if (Math.abs(v - val) > 1e-9f) return false;
        }
        return true;
    }
}
