package com.prelude.denoise;

import android.content.Context;
import android.util.Log;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import com.prelude.denoise.model.AssetModelSource;
import com.prelude.denoise.model.DenoisedFrame;
import com.prelude.denoise.model.FusedFrame;
import com.prelude.denoise.model.FrameGeometry;
import com.prelude.denoise.timeout.RaceResult;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * On-device smoke test of DenoiseService through the bundled asset
 * (decision A24). One 300x280 frame in BENCHMARK mode with the 500 ms
 * ceiling (uncalibrated — observations only, rule T4). Assertions are
 * structural only: no exception, output length, finite values.
 */
@RunWith(AndroidJUnit4.class)
public class DenoiseServiceDeviceTest {

    private static final String TAG = "DenoiseSvcDeviceTest";
    private static final String ASSET = "dncnn_int8_srgb.tflite";

    @Test
    public void denoiseOneFrameThroughBundledAsset() throws Exception {
        Context targetContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetModelSource source = new AssetModelSource(targetContext.getAssets(), ASSET);

        // SHA-256 computed at runtime from the loaded buffer (nothing hard-coded).
        ByteBuffer modelBuffer = source.loadModel();
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        modelBuffer.rewind();
        md.update(modelBuffer);
        String assetSha = toHex(md.digest());

        DenoiseService service = new DenoiseService(source, (msg, t) ->
            Log.w(TAG, msg, t == null ? null : t));
        try {
            int w = 300, h = 280, c = 3;
            float[] image = new float[w * h * c];
            Random rng = new Random(42); // fixed seed
            for (int i = 0; i < image.length; i++) {
                image[i] = rng.nextFloat();
            }

            RaceResult result = service.denoise(
                new FusedFrame("svc-device-1", "naive", image, "fusion_multi"),
                new FrameGeometry(w, h, c), 500L, DenoiseModule.RunMode.BENCHMARK);
            DenoisedFrame out = result.getFrame();

            assertEquals("output length", w * h * c, out.getImage().length);
            for (float v : out.getImage()) {
                assertTrue("output values must be finite", Float.isFinite(v));
            }

            JSONObject json = new JSONObject();
            json.put("deviceModel", android.os.Build.MODEL);
            json.put("androidVersion", android.os.Build.VERSION.RELEASE);
            json.put("timeoutCeilingMs", 500L);
            json.put("timeoutEvent", out.isTimeoutOccurred());
            json.put("discardRaceEvent", result.isDiscardRaceOccurred());
            json.put("latencyMs", out.getLatencyMs());
            json.put("assetSha256", assetSha);
            json.put("asset", ASSET);
            json.put("note", "uncalibrated 500 ms ceiling, BENCHMARK mode, observation only");

            File outFile = new File(targetContext.getFilesDir(),
                "denoise-service-device-" + System.currentTimeMillis() + ".json");
            try (FileOutputStream fos = new FileOutputStream(outFile)) {
                fos.write(json.toString(2).getBytes("UTF-8"));
            }
            Log.i(TAG, "DenoiseService device JSON written to " + outFile.getAbsolutePath());

            assertTrue("JSON must exist", outFile.exists() && outFile.length() > 0);
        } finally {
            service.close();
        }
    }

    private static String toHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
}
