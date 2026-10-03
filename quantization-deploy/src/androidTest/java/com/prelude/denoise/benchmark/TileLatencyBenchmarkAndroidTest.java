package com.prelude.denoise.benchmark;

import android.content.Context;
import android.os.Build;
import android.os.PowerManager;
import android.util.Log;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import com.prelude.denoise.model.LiteRtAdapter;
import com.prelude.denoise.model.LiteRtAdapterImpl;
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
 * Single-tile (256x256) on-device latency benchmark (rule D5).
 *
 * STATUS: NOT RUN on a physical device at time of writing — this file only
 * compiles (assembleDebugAndroidTest). Run with:
 *   cd capture-android && ./gradlew :quantization-deploy:connectedDebugAndroidTest
 *
 * Model source (rule C4):
 *   - Default: asset "dncnn-untrained-smoke.tflite" (untrained weights, sRGB
 *     smoke checkpoint). Latency of a fixed CNN architecture does not depend
 *     on weight values, so wiring-latency numbers from it are usable for the
 *     500 ms feasibility check, but they are marked harnessOnly=true in the
 *     JSON and must NEVER be submitted as results or reported as model quality.
 *   - Override: -Pandroid.testInstrumentationRunnerArguments.modelPath=<abs path>
 *     points at a real checkpoint pushed to the device (recommended:
 *     adb push <model.tflite> /sdcard/Android/data/<testPackage>/files/).
 *
 * Protocol: warm-up runs (excluded, count recorded) then N measured runs with
 * a fixed spacing between runs (thermal hygiene, rule D5); nearest-rank p95 in
 * fractional ms (same definition as the server); thermal status read before /
 * after warm-up / after the measured phase; device model, Android version and
 * SoC recorded; NUM_THREADS and the requested delegate recorded from
 * LiteRtAdapterImpl itself (not re-typed here).
 */
@RunWith(AndroidJUnit4.class)
public class TileLatencyBenchmarkAndroidTest {

    private static final String TAG = "TileLatencyBench";

    /** Tile edge in pixels; must match the converted model's fixed input (1x3x256x256). */
    private static final int TILE_SIZE = 256;
    /** Channels; must match the converted model (3, see training/model.py + convert_smoke.py). */
    private static final int CHANNELS = 3;
    /** Warm-up runs excluded from statistics; the count is recorded in the JSON. */
    private static final int WARMUP_RUNS = 5;
    /** Measured runs; p95 nearest-rank = ceil(0.95*N)-th smallest. */
    private static final int MEASURED_RUNS = 20;
    /** Pause between measured runs so consecutive runs do not self-heat the SoC (rule D5). */
    private static final long INTER_RUN_SPACING_MS = 100L;
    private static final String DEFAULT_ASSET = "dncnn-untrained-smoke.tflite";

    @Test
    public void singleTileLatencyBenchmark() throws Exception {
        Context targetContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        String modelPathArg = InstrumentationRegistry.getArguments().getString("modelPath");
        boolean harnessOnly;
        String modelRef;

        ByteBuffer modelBuffer;
        if (modelPathArg != null && !modelPathArg.trim().isEmpty()) {
            File f = new File(modelPathArg.trim());
            assertTrue("modelPath not readable on device: " + modelPathArg
                + " (adb push it to /sdcard/Android/data/<testPackage>/files/ and pass that path)",
                f.canRead());
            try (FileInputStream fis = new FileInputStream(f); FileChannel ch = fis.getChannel()) {
                modelBuffer = ByteBuffer.allocateDirect((int) f.length()).order(ByteOrder.nativeOrder());
                ch.read(modelBuffer);
                modelBuffer.rewind();
            }
            harnessOnly = f.getName().contains("untrained");
            modelRef = f.getName();
        } else {
            try (InputStream is = InstrumentationRegistry.getInstrumentation().getContext()
                .getAssets().open(DEFAULT_ASSET)) {
                ByteBuffer buffer = ByteBuffer.allocateDirect(is.available()).order(ByteOrder.nativeOrder());
                byte[] chunk = new byte[8192];
                int read;
                while ((read = is.read(chunk)) > 0) {
                    buffer.put(chunk, 0, read);
                }
                buffer.rewind();
                modelBuffer = buffer;
            }
            harnessOnly = true;
            modelRef = DEFAULT_ASSET + " (asset, untrained)";
        }

        // Capture the delegate/thread metadata LiteRtAdapterImpl itself reports.
        final List<String> infoLines = new ArrayList<>();
        LiteRtAdapter adapter = new LiteRtAdapterImpl(modelBuffer, (msg, t) -> infoLines.add(msg));

        int bufferBytes = TILE_SIZE * TILE_SIZE * CHANNELS * 4; // 1x3x256x256 float32
        ByteBuffer input = ByteBuffer.allocateDirect(bufferBytes).order(ByteOrder.nativeOrder());
        ByteBuffer output = ByteBuffer.allocateDirect(bufferBytes).order(ByteOrder.nativeOrder());
        // Synthetic tile: mid-gray with a mild diagonal gradient. Content does not
        // affect CNN latency; recorded in the JSON as synthetic.
        java.nio.FloatBuffer inputF = input.asFloatBuffer();
        for (int y = 0; y < TILE_SIZE; y++) {
            for (int x = 0; x < TILE_SIZE; x++) {
                for (int c = 0; c < CHANNELS; c++) {
                    inputF.put(0.5f + ((x + y) % 32) / 128.0f);
                }
            }
        }
        input.rewind();

        PowerManager pm = (PowerManager) targetContext.getSystemService(Context.POWER_SERVICE);
        int thermalBefore = thermalStatus(pm);
        assertNotNull("PowerManager unavailable", pm);

        for (int i = 0; i < WARMUP_RUNS; i++) {
            input.rewind();
            output.clear();
            adapter.run(input, output);
        }
        int thermalAfterWarmup = thermalStatus(pm);

        long[] latencyNs = new long[MEASURED_RUNS];
        for (int i = 0; i < MEASURED_RUNS; i++) {
            if (i > 0) {
                Thread.sleep(INTER_RUN_SPACING_MS);
            }
            input.rewind();
            output.clear();
            long startNs = System.nanoTime();
            adapter.run(input, output);
            latencyNs[i] = System.nanoTime() - startNs;
        }
        int thermalAfter = thermalStatus(pm);

        adapter.close();

        long[] sorted = latencyNs.clone();
        Arrays.sort(sorted);
        int p95Index = (int) Math.ceil(0.95 * MEASURED_RUNS) - 1;
        double p95Ms = sorted[Math.max(0, Math.min(p95Index, MEASURED_RUNS - 1))] / 1_000_000.0;
        double minMs = sorted[0] / 1_000_000.0;
        double maxMs = sorted[MEASURED_RUNS - 1] / 1_000_000.0;
        double medianMs = sorted[MEASURED_RUNS / 2] / 1_000_000.0;

        JSONObject json = new JSONObject();
        json.put("kind", "denoise-tile-latency");
        json.put("harnessOnly", harnessOnly);
        json.put("model", modelRef);
        json.put("input", "synthetic float tile (content-independent for latency)");
        json.put("inputShapeNchw", "1x" + CHANNELS + "x" + TILE_SIZE + "x" + TILE_SIZE + " float32");
        json.put("tileSizePx", TILE_SIZE);
        json.put("channels", CHANNELS);
        json.put("numThreads", LiteRtAdapterImpl.NUM_THREADS);
        json.put("delegate", "XNNPACK requested");
        json.put("adapterInfoLines", new JSONArray(infoLines));
        json.put("deviceModel", Build.MODEL);
        json.put("device", Build.DEVICE);
        json.put("androidVersion", Build.VERSION.RELEASE);
        json.put("sdkInt", Build.VERSION.SDK_INT);
        json.put("socModel", Build.SOC_MODEL);       // API 31+, minSdk 31
        json.put("hardware", Build.HARDWARE);
        json.put("warmupRuns", WARMUP_RUNS);
        json.put("measuredRuns", MEASURED_RUNS);
        json.put("interRunSpacingMs", INTER_RUN_SPACING_MS);
        json.put("thermalStatusBefore", thermalBefore);
        json.put("thermalStatusAfterWarmup", thermalAfterWarmup);
        json.put("thermalStatusAfter", thermalAfter);
        json.put("thermalScale", "PowerManager: 0=NONE 1=LIGHT 2=MODERATE 3=SEVERE 4=Critical 5=Emergency 6=Shutdown");
        JSONArray latencies = new JSONArray();
        for (long ns : latencyNs) {
            latencies.put(ns / 1_000_000.0);
        }
        json.put("latencyMs", latencies);
        json.put("minLatencyMs", minMs);
        json.put("medianLatencyMs", medianMs);
        json.put("maxLatencyMs", maxMs);
        json.put("p95LatencyMs", p95Ms);
        json.put("p95Method", "nearest-rank (ceil(0.95*N)-th smallest)");
        json.put("recordedAtUtc", java.time.Instant.now().toString());
        json.put("note", "If harnessOnly=true the model has untrained weights: latency numbers are "
            + "wiring-only evidence, never a reportable result (rule C4).");

        File dir = targetContext.getFilesDir();
        File outFile = new File(dir, "denoise-tile-latency-" + System.currentTimeMillis() + ".json");
        try (FileOutputStream fos = new FileOutputStream(outFile)) {
            fos.write(json.toString(2).getBytes("UTF-8"));
        }

        Log.i(TAG, "Latency JSON written to " + outFile.getAbsolutePath()
            + " p95=" + p95Ms + " ms (harnessOnly=" + harnessOnly + ")");

        // Only structural assertions: no hard-coded metric thresholds (no fabricated bars).
        assertTrue("Output JSON file must exist and be non-empty",
            outFile.exists() && outFile.length() > 0);
        JSONObject reread = new JSONObject(
            new String(java.nio.file.Files.readAllBytes(outFile.toPath()), "UTF-8"));
        assertTrue("p95 must be positive", reread.getDouble("p95LatencyMs") > 0.0);
        assertTrue("Must record excluded warm-up count", reread.getInt("warmupRuns") == WARMUP_RUNS);
    }

    private static int thermalStatus(PowerManager pm) {
        // POWER_SERVICE thermal API exists since API 30; minSdk is 31.
        return pm != null ? pm.getCurrentThermalStatus() : -1;
    }
}
