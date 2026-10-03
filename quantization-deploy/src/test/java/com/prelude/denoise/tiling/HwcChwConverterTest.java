package com.prelude.denoise.tiling;

import org.junit.Test;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class HwcChwConverterTest {

    @Test
    public void testRoundTrip() {
        int w = 2, h = 2, c = 3;
        float[] hwc = {
            1f, 2f, 3f,   4f, 5f, 6f,
            7f, 8f, 9f,  10f, 11f, 12f
        };
        float[] chw = new float[hwc.length];
        float[] hwcRoundTrip = new float[hwc.length];

        HwcChwConverter.hwcToChw(hwc, w, h, c, chw);
        HwcChwConverter.chwToHwc(chw, w, h, c, hwcRoundTrip);

        assertArrayEquals(hwc, hwcRoundTrip, 1e-6f);
    }

    @Test
    public void testAsymmetricPixelPattern() {
        int w = 3, h = 3, c = 3;
        float[] hwc = new float[w * h * c];
        // Set red channel (c=0) of top-right pixel (x=2, y=0)
        // HWC index = (y * w + x) * c + ch = (0 * 3 + 2) * 3 + 0 = 6
        hwc[6] = 99f;
        
        float[] chw = new float[w * h * c];
        HwcChwConverter.hwcToChw(hwc, w, h, c, chw);

        // In CHW, channel 0 is first block. Spatial index = y * w + x = 2
        // CHW index = ch * (w * h) + spatialIdx = 0 * 9 + 2 = 2
        assertEquals(99f, chw[2], 1e-6f);
        
        // Assert other values are 0
        for (int i = 0; i < chw.length; i++) {
            if (i != 2) {
                assertEquals(0f, chw[i], 1e-6f);
            }
        }
    }
}
