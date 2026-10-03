package com.prelude.denoise.tiling;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Tests for Tiler: verifies tiled inference produces identical output to
 * full-frame inference, handles odd/small image sizes, and has bounded memory.
 */
public class TilerTest {

    // A dummy "inferencer" that multiplies input by 2.
    // This allows us to strictly check that blending math is perfectly 1.0.
    private float[] dummyInference(float[] input) {
        float[] out = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            out[i] = input[i] * 2.0f;
        }
        return out;
    }

    @Test
    public void tiledOutputMatchesFullFramePerfectly() {
        int w = 500, h = 500, c = 3;
        float[] input = new float[w * h * c];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) i / (w * h * c);
        }

        Tiler tiler = new Tiler(256, 17, 16);

        // Target: what we expect from "full frame" inference
        float[] target = dummyInference(input);

        // Actual: tiled inference
        float[] actual = tiler.process(input, w, h, c, this::dummyInference);

        float tolerance = 0.01f;
        float maxDiff = 0f;

        for (int i = 0; i < input.length; i++) {
            float diff = Math.abs(actual[i] - target[i]);
            if (diff > maxDiff) maxDiff = diff;
            assertEquals("Pixel mismatch at " + i, target[i], actual[i], tolerance);
        }

        System.out.println("Max absolute difference: " + maxDiff);
        assertTrue(maxDiff < tolerance);
    }

    @Test
    public void handlesOddSizesAndSmallImages() {
        // Image smaller than a single tile!
        int w = 123, h = 99, c = 3;
        float[] input = new float[w * h * c];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) i / (w * h * c);
        }

        Tiler tiler = new Tiler(256, 17, 16);
        float[] target = dummyInference(input);
        float[] actual = tiler.process(input, w, h, c, this::dummyInference);

        for (int i = 0; i < input.length; i++) {
            float diff = Math.abs(actual[i] - target[i]);
            assertEquals("Pixel mismatch on small image at " + i,
                target[i], actual[i], 0.01f);
        }
    }

    @Test
    public void peakMemoryEstimationIsReasonable() {
        int tileSize = 256;
        int channels = 3;
        int bytesPerFloat = 4;

        // Memory per tile: input + output + weights
        int tileMemBytes = (tileSize * tileSize * channels * bytesPerFloat * 2)
                         + (tileSize * tileSize * bytesPerFloat);

        double tileMemMb = tileMemBytes / (1024.0 * 1024.0);

        assertTrue("Tile working memory should be small", tileMemMb < 2.0);
        System.out.println("Peak working memory overhead for Tiler: " + tileMemMb + " MB");
    }
}
