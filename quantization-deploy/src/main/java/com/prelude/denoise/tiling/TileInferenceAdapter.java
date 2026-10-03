package com.prelude.denoise.tiling;

import com.prelude.denoise.model.LiteRtAdapter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Adapts a LiteRtAdapter (ByteBuffer NCHW model) into a Tiler.Inferencer (float[] HWC tiles).
 * Performs HWC→NCHW conversion before inference and NCHW→HWC after.
 * Reuses direct ByteBuffers across tiles (no per-tile allocation).
 */
public final class TileInferenceAdapter implements Tiler.Inferencer {

    private final LiteRtAdapter adapter;
    private final int tileSize;
    private final int channels;
    private final ByteBuffer inputBuffer;
    private final ByteBuffer outputBuffer;

    public TileInferenceAdapter(LiteRtAdapter adapter, int tileSize, int channels) {
        this.adapter = adapter;
        this.tileSize = tileSize;
        this.channels = channels;
        int bytes = tileSize * tileSize * channels * 4;
        this.inputBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder());
        this.outputBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder());
    }

    @Override
    public float[] run(float[] tile) {
        int n = tileSize * tileSize * channels;
        float[] chw = new float[n];
        HwcChwConverter.hwcToChw(tile, tileSize, tileSize, channels, chw);

        inputBuffer.clear();
        inputBuffer.asFloatBuffer().put(chw);
        inputBuffer.rewind();
        outputBuffer.clear();

        adapter.run(inputBuffer, outputBuffer);

        outputBuffer.rewind();
        float[] chwOut = new float[n];
        outputBuffer.asFloatBuffer().get(chwOut);

        float[] hwcOut = new float[n];
        HwcChwConverter.chwToHwc(chwOut, tileSize, tileSize, channels, hwcOut);
        return hwcOut;
    }
}
