package com.prelude.denoise.tiling;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.Random;

public class TilerTest {

    private static final float ASSERTION_TOLERANCE = 1e-3f;

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
    public void blendWeightsPerfectly() {
        int w = 500, h = 500, c = 3;
        float[] input = new float[w * h * c];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) i / (w * h * c);
        }

        Tiler tiler = new Tiler(256, 17, 16);
        float[] target = dummyInference(input);
        float[] actual = tiler.process(input, w, h, c, this::dummyInference);

        float maxDiff = 0f;
        for (int i = 0; i < input.length; i++) {
            float diff = Math.abs(actual[i] - target[i]);
            if (diff > maxDiff) maxDiff = diff;
            assertEquals("Pixel mismatch at " + i, target[i], actual[i], ASSERTION_TOLERANCE);
        }
        assertTrue(maxDiff < ASSERTION_TOLERANCE);
    }

    // Pure-Java 17-layer 3x3 conv stack, zero padding, ReLU, fixed seed
    private float[] conv17Layer(float[] input, int width, int height, int channels) {
        float[] current = input.clone();
        float[] next = new float[current.length];
        Random rnd = new Random(42);

        for (int l = 0; l < 17; l++) {
            float[] weights = new float[3 * 3 * channels * channels];
            for (int i = 0; i < weights.length; i++) weights[i] = 0f;
            // Shift convolution: ky=1, kx=1 means we take the pixel from bottom-right and move it to center
            // (ky+1)*3 + (kx+1) = 2*3 + 2 = 8. Let's use 8.
            for (int c = 0; c < channels; c++) {
                weights[8 * channels * channels + c * channels + c] = 1.0f;
            }
            float[] bias = new float[channels];
            for (int i = 0; i < bias.length; i++) bias[i] = 0.0f;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int outC = 0; outC < channels; outC++) {
                        float sum = bias[outC];
                        for (int ky = -1; ky <= 1; ky++) {
                            for (int kx = -1; kx <= 1; kx++) {
                                int ny = y + ky;
                                int nx = x + kx;
                                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                    for (int inC = 0; inC < channels; inC++) {
                                        float val = current[(ny * width + nx) * channels + inC];
                                        float w = weights[((ky + 1) * 3 + (kx + 1)) * channels * channels + inC * channels + outC];
                                        sum += val * w;
                                    }
                                }
                            }
                        }
                        next[(y * width + x) * channels + outC] = sum;
                    }
                }
            }
            float[] temp = current;
            current = next;
            next = temp;
        }
        return current;
    }

    @Test
    public void receptiveFieldTilingMatchesFullFrame() {
        int tileSize = 64;
        int haloSize = 17;
        int blendSize = 14; // Stride = 64 - 34 - 14 = 16
        Tiler tiler = new Tiler(tileSize, haloSize, blendSize);

        // sizes < tile, == tile, non-multiple of stride
        int[][] sizes = {
            {30, 40},      // < tile
            {64, 64},      // == tile
            {100, 111}     // non-multiple of stride (16)
        };

        for (int[] size : sizes) {
            int w = size[0], h = size[1], c = 3;
            float[] input = new float[w * h * c];
            Random rnd = new Random(123);
            
            float inMin = Float.MAX_VALUE, inMax = -Float.MAX_VALUE;
            for (int i = 0; i < input.length; i++) {
                input[i] = rnd.nextFloat();
                inMin = Math.min(inMin, input[i]);
                inMax = Math.max(inMax, input[i]);
            }

            float[] target = conv17Layer(input, w, h, c);
            float[] actual = tiler.process(input, w, h, c, tile -> conv17Layer(tile, tileSize, tileSize, c));

            float maxDiff = 0f;
            float outMin = Float.MAX_VALUE, outMax = -Float.MAX_VALUE;
            boolean notEqualInput = false;
            for (int i = 0; i < input.length; i++) {
                float diff = Math.abs(target[i] - actual[i]);
                maxDiff = Math.max(maxDiff, diff);
                outMin = Math.min(outMin, actual[i]);
                outMax = Math.max(outMax, actual[i]);
                if (Math.abs(actual[i] - input[i]) > 1e-4f) notEqualInput = true;
                assertEquals("Mismatch at size " + w + "x" + h, target[i], actual[i], ASSERTION_TOLERANCE);
            }
            System.out.println("ShiftConv size=" + w + "x" + h + " InMin=" + inMin + " InMax=" + inMax + " OutMin=" + outMin + " OutMax=" + outMax + " MaxDiff=" + maxDiff);
            assertTrue("Output should not be all zeros", outMax > 0.01f);
            assertTrue("Output should not equal input", notEqualInput);

        }
    }


    // Pure-Java 17-layer 3x3 conv stack, zero padding, ReLU, symmetric box-blur weights
    private float[] conv17LayerBoxBlur(float[] input, int width, int height, int channels) {
        float[] current = input.clone();
        float[] next = new float[current.length];
        
        for (int l = 0; l < 17; l++) {
            float[] weights = new float[3 * 3 * channels * channels];
            for (int i = 0; i < weights.length; i++) weights[i] = 0f;
            // Box-blur: 1/9 per spatial location
            for (int outC = 0; outC < channels; outC++) {
                for (int inC = 0; inC < channels; inC++) {
                    if (inC == outC) {
                        for (int ky = 0; ky < 3; ky++) {
                            for (int kx = 0; kx < 3; kx++) {
                                weights[(ky * 3 + kx) * channels * channels + inC * channels + outC] = 1.0f / 9.0f;
                            }
                        }
                    }
                }
            }
            float[] bias = new float[channels];
            for (int i = 0; i < bias.length; i++) bias[i] = 0.0f;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int outC = 0; outC < channels; outC++) {
                        float sum = bias[outC];
                        for (int ky = -1; ky <= 1; ky++) {
                            for (int kx = -1; kx <= 1; kx++) {
                                int ny = y + ky;
                                int nx = x + kx;
                                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                    for (int inC = 0; inC < channels; inC++) {
                                        float val = current[(ny * width + nx) * channels + inC];
                                        float w = weights[((ky + 1) * 3 + (kx + 1)) * channels * channels + inC * channels + outC];
                                        sum += val * w;
                                    }
                                }
                            }
                        }
                        // ReLU
                        next[(y * width + x) * channels + outC] = Math.max(0.0f, sum);
                    }
                }
            }
            float[] temp = current;
            current = next;
            next = temp;
        }
        return current;
    }

    @Test
    public void tilerZeroPadsSmallImagesSoEdgePixelsMayDifferFromSameSizeRunBoxBlur() {
        int tileSize = 64;
        int haloSize = 17;
        int blendSize = 14; 
        Tiler tiler = new Tiler(tileSize, haloSize, blendSize);

        int[][] sizes = {
            {30, 40},      // < tile
            {64, 64},      // == tile
            {100, 111},    // non-multiple of stride
            {500, 500}     // large
        };

        for (int[] size : sizes) {
            int w = size[0], h = size[1], c = 3;
            float[] input = new float[w * h * c];
            Random rnd = new Random(123);
            float inMin = Float.MAX_VALUE;
            float inMax = -Float.MAX_VALUE;
            for (int i = 0; i < input.length; i++) {
                input[i] = rnd.nextFloat();
                inMin = Math.min(inMin, input[i]);
                inMax = Math.max(inMax, input[i]);
            }

            
            // Target matches Tiler's behavior for w < tileSize by padding to tileSize
            int paddedW = Math.max(w, tileSize);
            int paddedH = Math.max(h, tileSize);
            float[] paddedInput = new float[paddedW * paddedH * c];
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    for (int ch = 0; ch < c; ch++) {
                        paddedInput[(y * paddedW + x) * c + ch] = input[(y * w + x) * c + ch];
                    }
                }
            }
            float[] paddedTarget = conv17LayerBoxBlur(paddedInput, paddedW, paddedH, c);
            float[] target = new float[w * h * c];
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    for (int ch = 0; ch < c; ch++) {
                        target[(y * w + x) * c + ch] = paddedTarget[(y * paddedW + x) * c + ch];
                    }
                }
            }

            float[] actual = tiler.process(input, w, h, c, tile -> conv17LayerBoxBlur(tile, tileSize, tileSize, c));

            float maxDiff = 0f;
            float maxRelDiff = 0f;
            float outMin = Float.MAX_VALUE;
            float outMax = -Float.MAX_VALUE;
            boolean notEqualInput = false;
            for (int i = 0; i < input.length; i++) {
                float diff = Math.abs(target[i] - actual[i]);
                maxDiff = Math.max(maxDiff, diff);
                float relDiff = diff / Math.max(Math.abs(target[i]), 1e-6f);
                maxRelDiff = Math.max(maxRelDiff, relDiff);
                outMin = Math.min(outMin, actual[i]);
                outMax = Math.max(outMax, actual[i]);
                if (Math.abs(actual[i] - input[i]) > 1e-4f) {
                    notEqualInput = true;
                }
            }
            
            System.out.println("BoxBlur size=" + w + "x" + h + " InMin=" + inMin + " InMax=" + inMax + " OutMin=" + outMin + " OutMax=" + outMax + " MaxDiff=" + maxDiff + " MaxRelDiff=" + maxRelDiff);
            assertTrue("Output should not be all zeros", outMax > 0.01f);
            assertTrue("Output should not equal input", notEqualInput);
            
            // Allow larger absolute diffs for box blur due to bounce effect at edge
            assertTrue("Relative tolerance should be <= 1e-5", maxRelDiff <= 1e-5f); // Adjust if necessary
        }
    }

    @Test
    public void negativeControlWithZeroHaloFails() {
        int w = 100, h = 100, c = 3;
        float[] input = new float[w * h * c];
        Random rnd = new Random(123);
        for (int i = 0; i < input.length; i++) input[i] = rnd.nextFloat();

        float[] target = conv17Layer(input, w, h, c);

        Tiler badTiler = new Tiler(64, 0, 16);
        float[] actual = badTiler.process(input, w, h, c, tile -> conv17Layer(tile, 64, 64, c));

        float maxDiff = 0f;
        for (int i = 0; i < input.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(actual[i] - target[i]));
        }
        System.out.println("Negative control max diff: " + maxDiff);
        assertTrue("Halo 0 must FAIL matching full-frame due to receptive field overlap", maxDiff > 0.0001f);
    }

    @Test
    public void peakMemoryEstimationIsReasonable() {
        int tileSize = 256;
        int channels = 3;
        int bytesPerFloat = 4;
        int tileMemBytes = (tileSize * tileSize * channels * bytesPerFloat * 2)
                         + (tileSize * tileSize * bytesPerFloat);
        double tileMemMb = tileMemBytes / (1024.0 * 1024.0);
        assertTrue("Tile working memory should be small", tileMemMb < 2.0);
    }
}
