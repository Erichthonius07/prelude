package com.prelude.denoise.tiling;

import com.prelude.denoise.model.LiteRtAdapter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Adapts a LiteRtAdapter (ByteBuffer NCHW model) into a Tiler.Inferencer (float[] HWC tiles).
 * Performs HWC→NCHW conversion before inference and NCHW→HWC after.
 *
 * Reuses the direct ByteBuffers and the conversion scratch arrays across tiles:
 * no per-tile heap allocation, so no GC pauses land inside the measured latency.
 *
 * NOT thread-safe by design: one instance drives one tiler.process loop on one
 * inference thread, and the array returned by run() is reused by the next call.
 * The Tiler consumes the returned tile synchronously before invoking run() again,
 * so reuse is safe; the array must never be stored beyond that call.
 */
public final class TileInferenceAdapter implements Tiler.Inferencer {

    private final LiteRtAdapter adapter;
    private final int tileSize;
    private final int channels;
    private final ByteBuffer inputBuffer;
    private final ByteBuffer outputBuffer;
    private final float[] chw;
    private final float[] chwOut;
    private final float[] hwcOut;

    public TileInferenceAdapter(LiteRtAdapter adapter, int tileSize, int channels) {
        this.adapter = adapter;
        this.tileSize = tileSize;
        this.channels = channels;
        int bytes = tileSize * tileSize * channels * 4;
        this.inputBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder());
        this.outputBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder());
        int floats = tileSize * tileSize * channels;
        this.chw = new float[floats];
        this.chwOut = new float[floats];
        this.hwcOut = new float[floats];
    }

    @Override
    public float[] run(float[] tile) {
        HwcChwConverter.hwcToChw(tile, tileSize, tileSize, channels, chw);

        inputBuffer.clear();
        inputBuffer.asFloatBuffer().put(chw);
        inputBuffer.rewind();
        outputBuffer.clear();

        adapter.run(inputBuffer, outputBuffer);

        outputBuffer.rewind();
        outputBuffer.asFloatBuffer().get(chwOut);

        HwcChwConverter.chwToHwc(chwOut, tileSize, tileSize, channels, hwcOut);
        return hwcOut;
    }
}
