package com.prelude.denoise.tiling;

public class HwcChwConverter {
    public static void hwcToChw(float[] hwc, int w, int h, int c, float[] chw) {
        int spatialSize = w * h;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int spatialIdx = y * w + x;
                for (int ch = 0; ch < c; ch++) {
                    chw[ch * spatialSize + spatialIdx] = hwc[spatialIdx * c + ch];
                }
            }
        }
    }

    public static void chwToHwc(float[] chw, int w, int h, int c, float[] hwc) {
        int spatialSize = w * h;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int spatialIdx = y * w + x;
                for (int ch = 0; ch < c; ch++) {
                    hwc[spatialIdx * c + ch] = chw[ch * spatialSize + spatialIdx];
                }
            }
        }
    }
}
