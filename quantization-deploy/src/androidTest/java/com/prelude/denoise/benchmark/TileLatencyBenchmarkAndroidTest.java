package com.prelude.denoise.benchmark;

import android.content.Context;
import android.os.Build;
import android.os.PowerManager;
import android.util.Log;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import com.prelude.denoise.DenoiseModule;
import com.prelude.denoise.model.FusedFrame;
import com.prelude.denoise.model.FrameGeometry;
import com.prelude.denoise.model.LiteRtAdapter;
import com.prelude.denoise.model.LiteRtAdapterImpl;
import com.prelude.denoise.timeout.ErrorListener;
import com.prelude.denoise.timeout.RaceResult;
import com.prelude.denoise.tiling.Tiler;
import com.prelude.denoise.tiling.TileInferenceAdapter;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * On-device latency benchmark (rule D5). STATUS: NOT RUN on a physical device —
 * this file only compiles (assembleDebugAndroidTest). Run with:
 *   cd capture-android && ./gradlew :quantization-deploy:connectedDebugAndroidTest
 *
 * Instrumentation arguments (all optional):
 *   measuredRuns  — single-tile measured runs,            default 100
 *   warmupRuns    — single-tile warm-up runs (excluded),  default 5
 *   imageSize     — optional "<w>x<h>": ALSO benchmark a full tiled denoise
 *                   of that size, repeats times (default 3), with a cool-down
 *                   between repeats
 *   repeats       — full-image repeats,                   default 3
 *   cooldownMs    — pause between full-image repeats,     default 2000
 *   modelPath     — absolute device path to a .tflite checkpoint (overrides the
 *                   bundled untrained asset)
 *
 * Model source (rule C4 as amended): the default bundled asset is an untrained
 * smoke checkpoint. Latency-only timing of untrained models is allowed with
 * JSON flag latencyOnly=true and file names containing "untrained" or "smoke";
 * NEVER quality, NEVER a final result. A real checkpoint (modelPath) sets
 * latencyOnly=false.
 *
 * Thermal policy (rule D5): refuse to start at
 * PowerManager.THERMAL_STATUS_MODERATE or above; abort the phase and write
 * partial results if MODERATE is reached mid-run; status recorded before, every
 * 10 measured runs, and after each phase; thermalRose=true if any reading
 * exceeded the starting status.
 */
@RunWith(AndroidJUnit4.class)
public class TileLatencyBenchmarkAndroidTest {

    private static final String TAG = "TileLatencyBench";

    /** Tile edge in pixels; must match the converted model's fixed input (1x3x256x256). */
    private static final int TILE_SIZE = 256;
    /** Channels; must match the converted model (3, see training/model.py + convert_smoke.py). */
    private static final int CHANNELS = 3;
    private static final int DEFAULT_MEASURED_RUNS = 100;
    private static final int DEFAULT_WARMUP_RUNS = 5;
    private static final int DEFAULT_REPEATS = 3;
    private static final long DEFAULT_COOLDOWN_MS = 2000L;
    private static final long INTER_RUN_SPACING_MS = 100L;
    private static final long UNCALIBRATED_TIMEOUT_MS = 500L; // T4 ceiling (loud log below)
    private static final String DEFAULT_ASSET = "dncnn-untrained-smoke.tflite";

    private PowerManager powerManager;
    private int thermalStart;
    private boolean thermalRose = false;

    @Test
    public void latencyBenchmark() throws Exception {
        android.os.Bundle args = InstrumentationRegistry.getArguments();
        int measuredRuns = intArg(args, "measuredRuns", DEFAULT_MEASURED_RUNS);
        int warmupRuns = intArg(args, "warmupRuns", DEFAULT_WARMUP_RUNS);
        int repeats = intArg(args, "repeats", DEFAULT_REPEATS);
        long cooldownMs = longArg(args, "cooldownMs", DEFAULT_COOLDOWN_MS);
        String imageSizeArg = args.getString("imageSize");
        String modelPathArg = args.getString("modelPath");

        Context targetContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        powerManager = (PowerManager) targetContext.getSystemService(Context.POWER_SERVICE);
        assertNotNull("PowerManager unavailable", powerManager);

        // Thermal gate: refuse to start at MODERATE or above (rule D5).
        thermalStart = powerManager.getCurrentThermalStatus();
        if (thermalStart >= PowerManager.THERMAL_STATUS_MODERATE) {
            fail("Refusing to benchmark: device already at thermal status " + thermalStart
                + " (>= MODERATE). Let it cool down and re-run. (rule D5)");
            return;
        }

        ByteBuffer modelBuffer = loadModel(modelPathArg);
        boolean latencyOnly = modelPathArg == null || new File(modelPathArg.trim()).getName().contains("untrained");

        List<String> infoLines = new ArrayList<>();
        LiteRtAdapter adapter = new LiteRtAdapterImpl(modelBuffer, new ErrorListener() {
            @Override
            public void onError(String msg, Throwable t) {
                Log.w(TAG, "LiteRtAdapterImpl: " + msg, t);
            }
            @Override
            public void onInfo(String msg) {
                infoLines.add(msg);
                Log.i(TAG, msg);
            }
        });

        JSONObject json = new JSONObject();
        json.put("kind", "denoise-latency");
        json.put("latencyOnly", latencyOnly);
        json.put("model", modelPathArg == null ? DEFAULT_ASSET + " (asset, untrained)"
            : new File(modelPathArg.trim()).getName());
        json.put("numThreads", LiteRtAdapterImpl.NUM_THREADS);
        json.put("delegate", "XNNPACK requested");
        json.put("adapterInfoLines", new JSONArray(infoLines));
        json.put("deviceModel", Build.MODEL);
        json.put("device", Build.DEVICE);
        json.put("androidVersion", Build.VERSION.RELEASE);
        json.put("sdkInt", Build.VERSION.SDK_INT);
        json.put("socModel", Build.SOC_MODEL);       // API 31+, minSdk 31
        json.put("hardware", Build.HARDWARE);
        json.put("args", new JSONObject()
            .put("measuredRuns", measuredRuns)
            .put("warmupRuns", warmupRuns)
            .put("imageSize", imageSizeArg == null ? JSONObject.NULL : imageSizeArg)
            .put("repeats", repeats)
            .put("cooldownMs", cooldownMs));

        // ---- Phase 1: single-tile latency (always runs) ----
        json.put("singleTile", singleTilePhase(adapter, warmupRuns, measuredRuns));

        // ---- Phase 2: full-image tiled denoise (only with imageSize=<w>x<h>) ----
        if (imageSizeArg != null && !imageSizeArg.trim().isEmpty()) {
            String[] parts = imageSizeArg.trim().toLowerCase().split("x");
            assertTrue("imageSize must be <w>x<h>, got: " + imageSizeArg, parts.length == 2);
            int w = Integer.parseInt(parts[0].trim());
            int h = Integer.parseInt(parts[1].trim());
            json.put("fullImage", fullImagePhase(adapter, w, h, repeats, cooldownMs));
        }

        json.put("thermalStart", thermalStart);
        json.put("thermalRose", thermalRose);
        json.put("thermalScale", "PowerManager: 0=NONE 1=LIGHT 2=MODERATE 3=SEVERE 4=Critical 5=Emergency 6=Shutdown");
        json.put("recordedAtUtc", java.time.Instant.now().toString());
        json.put("note", latencyOnly
            ? "latencyOnly=true: untrained-weight model (rule C4). Wiring latency evidence only — "
              + "never a quality result or a reportable benchmark number."
            : "Real checkpoint supplied via modelPath; still latency-only evidence (no ground truth here).");

        File outFile = writeJson(targetContext, json);
        Log.i(TAG, "Latency JSON written to " + outFile.getAbsolutePath());

        assertTrue("Output JSON file must exist and be non-empty",
            outFile.exists() && outFile.length() > 0);
    }

    // ------------------------------------------------------------------

    private JSONObject singleTilePhase(LiteRtAdapter adapter, int warmupRuns, int measuredRuns)
            throws Exception {
        int bufferBytes = TILE_SIZE * TILE_SIZE * CHANNELS * 4; // 1x3x256x256 float32
        ByteBuffer input = ByteBuffer.allocateDirect(bufferBytes).order(ByteOrder.nativeOrder());
        ByteBuffer output = ByteBuffer.allocateDirect(bufferBytes).order(ByteOrder.nativeOrder());
        java.nio.FloatBuffer inputF = input.asFloatBuffer();
        for (int y = 0; y < TILE_SIZE; y++) {
            for (int x = 0; x < TILE_SIZE; x++) {
                for (int c = 0; c < CHANNELS; c++) {
                    inputF.put(0.5f + ((x + y) % 32) / 128.0f);
                }
            }
        }

        for (int i = 0; i < warmupRuns; i++) {
            input.rewind();
            output.clear();
            adapter.run(input, output);
        }
        int thermalAfterWarmup = recordThermal();

        List<Double> latenciesMs = new ArrayList<>();
        boolean aborted = false;
        String abortReason = null;
        for (int i = 0; i < measuredRuns; i++) {
            // Thermal check every 10 runs: abort at MODERATE, keep partial results (D5).
            if (i % 10 == 0) {
                int status = recordThermal();
                if (status >= PowerManager.THERMAL_STATUS_MODERATE) {
                    aborted = true;
                    abortReason = "thermal status reached " + status + " (>= MODERATE) after "
                        + latenciesMs.size() + " measured runs";
                    Log.w(TAG, abortReason);
                    break;
                }
            }
            if (i > 0) {
                Thread.sleep(INTER_RUN_SPACING_MS);
            }
            input.rewind();
            output.clear();
            long startNs = System.nanoTime();
            adapter.run(input, output);
            latenciesMs.add((System.nanoTime() - startNs) / 1_000_000.0);
        }
        int thermalAfter = recordThermal();

        JSONObject phase = new JSONObject();
        phase.put("tileSizePx", TILE_SIZE);
        phase.put("channels", CHANNELS);
        phase.put("warmupRuns", warmupRuns);
        phase.put("measuredRunsRequested", measuredRuns);
        phase.put("runsCompleted", latenciesMs.size());
        phase.put("interRunSpacingMs", INTER_RUN_SPACING_MS);
        phase.put("thermalAfterWarmup", thermalAfterWarmup);
        phase.put("thermalAfter", thermalAfter);
        phase.put("latencyMs", new JSONArray(latenciesMs));
        phase.put("p95LatencyMs", nearestRankP95(latenciesMs));
        phase.put("p95Method", "nearest-rank (ceil(0.95*N)-th smallest over runsCompleted values)");
        phase.put("aborted", aborted);
        phase.put("abortReason", abortReason == null ? JSONObject.NULL : abortReason);
        return phase;
    }

    private JSONObject fullImagePhase(LiteRtAdapter adapter, int w, int h,
                                      int repeats, long cooldownMs) throws Exception {
        Tiler tiler = new Tiler();
        FrameGeometry geom = new FrameGeometry(w, h, CHANNELS);
        // Synthetic linear-domain-shaped tile content (latency is content-independent).
        float[] image = new float[w * h * CHANNELS];
        for (int i = 0; i < image.length; i++) {
            image[i] = (i % 251) / 251.0f;
        }

        ErrorListener moduleErrors = (msg, t) -> Log.w(TAG, "DenoiseModule: " + msg,
            t == null ? null : t);
        DenoiseModule module = new DenoiseModule(
            frame -> { throw new UnsupportedOperationException("tiled benchmark only"); },
            com.prelude.denoise.timeout.SystemTimeSource.INSTANCE,
            DenoiseModule.MODEL_VARIANT_CNN_V1,
            DenoiseModule.PRECISION_FP32,
            moduleErrors,
            tiler,
            adapter);

        // Counting + timing wrapper around the real adapter (single-threaded use only:
        // the uncapped phase and the DenoiseModule phase run strictly sequentially).
        List<Double> uncappedTileNativeMs = new ArrayList<>();
        final int[] uncappedTileCount = {0};
        LiteRtAdapter counting = new LiteRtAdapter() {
            @Override
            public void run(ByteBuffer in, ByteBuffer out) {
                long startNs = System.nanoTime();
                adapter.run(in, out);
                uncappedTileNativeMs.add((System.nanoTime() - startNs) / 1_000_000.0);
                uncappedTileCount[0]++;
            }
            @Override
            public void close() {
            }
        };
        TileInferenceAdapter tileAdapter = new TileInferenceAdapter(counting, TILE_SIZE, CHANNELS);
        // T9 abandon flag doubles as the thermal abort for the uncapped phase.
        java.util.concurrent.atomic.AtomicBoolean thermalAbort =
            new java.util.concurrent.atomic.AtomicBoolean(false);

        JSONArray repeatResults = new JSONArray();
        boolean phaseAborted = false;
        String phaseAbortReason = null;
        for (int r = 0; r < repeats; r++) {
            int status = recordThermal();
            if (status >= PowerManager.THERMAL_STATUS_MODERATE) {
                phaseAborted = true;
                phaseAbortReason = "thermal status " + status + " (>= MODERATE) before repeat " + r;
                Log.w(TAG, phaseAbortReason);
                break;
            }
            if (r > 0) {
                Thread.sleep(cooldownMs); // cool-down between repeats (rule D5)
            }

            JSONObject rep = new JSONObject();
            rep.put("repeat", r);

            // (a) Uncapped: tiler.process directly with the real adapter, no race,
            //     no timeout — the raw cost of the whole image.
            uncappedTileNativeMs.clear();
            uncappedTileCount[0] = 0;
            thermalAbort.set(false);
            long startNs = System.nanoTime();
            float[] uncappedOut = tiler.process(image, w, h, CHANNELS, tileAdapter,
                () -> {
                    if (recordThermal() >= PowerManager.THERMAL_STATUS_MODERATE) {
                        thermalAbort.set(true);
                    }
                    return thermalAbort.get();
                });
            double uncappedTotalMs = (System.nanoTime() - startNs) / 1_000_000.0;
            JSONObject uncapped = new JSONObject();
            uncapped.put("totalMs", uncappedTotalMs);
            uncapped.put("tileCount", uncappedTileCount[0]);
            uncapped.put("perTileNativeMs", new JSONArray(uncappedTileNativeMs));
            uncapped.put("abandoned", uncappedOut == null);
            uncapped.put("abandonReason", uncappedOut == null
                ? "thermal abort between tiles (T9 abandon flag)" : JSONObject.NULL);
            rep.put("uncapped", uncapped);

            // (b) Through DenoiseModule in BENCHMARK mode with the T4 500 ms ceiling.
            //     timeoutEvent=true here IS the feasibility signal for the ceiling.
            Log.w(TAG, "Full-image run uses UNCALIBRATED_TIMEOUT_MS=" + UNCALIBRATED_TIMEOUT_MS
                + " (rule T4: benchmark/graded runs need a calibrated profile)");
            RaceResult result = module.denoise(new FusedFrame(
                "bench-full-" + r, "naive", image.clone(), "fusion_multi"),
                geom, UNCALIBRATED_TIMEOUT_MS, DenoiseModule.RunMode.BENCHMARK);
            JSONObject through = new JSONObject();
            through.put("timeoutEvent", result.getFrame().isTimeoutOccurred());
            through.put("discardRaceEvent", result.isDiscardRaceOccurred());
            through.put("latencyMs", result.getFrame().getLatencyMs());
            through.put("fallbackReason", result.getFallbackReason() == null
                ? JSONObject.NULL : result.getFallbackReason());
            rep.put("throughDenoiseModule", through);

            repeatResults.put(rep);
        }

        JSONObject phase = new JSONObject();
        phase.put("imageSize", w + "x" + h);
        phase.put("repeatsRequested", repeats);
        phase.put("repeatsCompleted", repeatResults.length());
        phase.put("cooldownMs", cooldownMs);
        phase.put("timeoutCeilingMs", UNCALIBRATED_TIMEOUT_MS);
        phase.put("calibrated", false);
        phase.put("repeats", repeatResults);
        phase.put("aborted", phaseAborted);
        phase.put("abortReason", phaseAbortReason == null ? JSONObject.NULL : phaseAbortReason);
        return phase;
    }

    // ------------------------------------------------------------------

    private int recordThermal() {
        int status = powerManager.getCurrentThermalStatus();
        if (status > thermalStart) {
            thermalRose = true;
        }
        return status;
    }

    private static double nearestRankP95(List<Double> valuesMs) {
        if (valuesMs.isEmpty()) {
            return Double.NaN;
        }
        double[] sorted = new double[valuesMs.size()];
        for (int i = 0; i < sorted.length; i++) {
            sorted[i] = valuesMs.get(i);
        }
        Arrays.sort(sorted);
        int idx = (int) Math.ceil(0.95 * sorted.length) - 1;
        return sorted[Math.max(0, Math.min(idx, sorted.length - 1))];
    }

    private ByteBuffer loadModel(String modelPathArg) throws Exception {
        if (modelPathArg != null && !modelPathArg.trim().isEmpty()) {
            File f = new File(modelPathArg.trim());
            assertTrue("modelPath not readable on device: " + modelPathArg
                + " (adb push it to /sdcard/Android/data/<testPackage>/files/ and pass that path)",
                f.canRead());
            try (FileInputStream fis = new FileInputStream(f); FileChannel ch = fis.getChannel()) {
                ByteBuffer modelBuffer = ByteBuffer.allocateDirect((int) f.length())
                    .order(ByteOrder.nativeOrder());
                ch.read(modelBuffer);
                modelBuffer.rewind();
                return modelBuffer;
            }
        }
        try (InputStream is = InstrumentationRegistry.getInstrumentation().getContext()
            .getAssets().open(DEFAULT_ASSET)) {
            ByteBuffer buffer = ByteBuffer.allocateDirect(is.available()).order(ByteOrder.nativeOrder());
            byte[] chunk = new byte[8192];
            int read;
            while ((read = is.read(chunk)) > 0) {
                buffer.put(chunk, 0, read);
            }
            buffer.rewind();
            return buffer;
        }
    }

    private File writeJson(Context targetContext, JSONObject json) throws Exception {
        File dir = targetContext.getFilesDir();
        File outFile = new File(dir, "denoise-latency-" + System.currentTimeMillis() + ".json");
        try (FileOutputStream fos = new FileOutputStream(outFile)) {
            fos.write(json.toString(2).getBytes("UTF-8"));
        }
        return outFile;
    }

    private static int intArg(android.os.Bundle args, String key, int def) {
        String v = args.getString(key);
        return v == null ? def : Integer.parseInt(v.trim());
    }

    private static long longArg(android.os.Bundle args, String key, long def) {
        String v = args.getString(key);
        return v == null ? def : Long.parseLong(v.trim());
    }

    private static void fail(String message) {
        throw new AssertionError(message);
    }
}
