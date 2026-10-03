package com.prelude.denoise.model;

import com.prelude.denoise.timeout.InferenceRunner;
import com.prelude.denoise.timeout.ErrorListener;
import com.prelude.denoise.tiling.HwcChwConverter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class LiteRtInferenceRunner implements InferenceRunner, AutoCloseable {
    private final LiteRtAdapter adapter;
    private final ErrorListener errorListener;
    private final int width;
    private final int height;
    private final int channels;
    
    // Direct ByteBuffers for zero-copy JNI boundaries (reused)
    private final ByteBuffer inputBuffer;
    private final ByteBuffer outputBuffer;

    // Conversion scratch arrays, reused across runs to avoid GC pressure inside
    // the measured latency. The HWC output array is NOT reused: it escapes into
    // the delivered DenoisedFrame, so each run allocates it fresh — a later run
    // must never overwrite an already-delivered frame's pixels (§6.4).
    private final float[] chwInput;
    private final float[] chwOutput;
    
    // Concurrency control: one instance, never run concurrently (rule D4).
    private final Semaphore runLock = new Semaphore(1);
    private final AtomicBoolean isClosed = new AtomicBoolean(false);

    public LiteRtInferenceRunner(LiteRtAdapter adapter, ErrorListener errorListener, 
                                 int width, int height, int channels) {
        this.adapter = adapter;
        this.errorListener = errorListener;
        this.width = width;
        this.height = height;
        this.channels = channels;
        
        int bufferBytes = width * height * channels * 4; // float32
        this.inputBuffer = ByteBuffer.allocateDirect(bufferBytes).order(ByteOrder.nativeOrder());
        this.outputBuffer = ByteBuffer.allocateDirect(bufferBytes).order(ByteOrder.nativeOrder());
        int floats = width * height * channels;
        this.chwInput = new float[floats];
        this.chwOutput = new float[floats];
    }

    @Override
    public float[] run(FusedFrame input) {
        if (isClosed.get()) {
            throw new IllegalStateException("Runner is closed");
        }

        if (!runLock.tryAcquire()) {
            throw new IllegalStateException("Interpreter called concurrently");
        }

        try {
            float[] hwcInput = input.getImage();

            // HWC -> NCHW translation (into reused scratch)
            HwcChwConverter.hwcToChw(hwcInput, width, height, channels, chwInput);

            // Copy to direct buffer
            inputBuffer.clear();
            inputBuffer.asFloatBuffer().put(chwInput);
            inputBuffer.rewind();

            outputBuffer.clear();

            adapter.run(inputBuffer, outputBuffer);

            outputBuffer.rewind();
            outputBuffer.asFloatBuffer().get(chwOutput);

            float[] hwcOutput = new float[chwOutput.length];
            // NCHW -> HWC translation
            HwcChwConverter.chwToHwc(chwOutput, width, height, channels, hwcOutput);

            return hwcOutput;
        } finally {
            runLock.release();
        }
    }

    @Override
    public void close() {
        if (isClosed.compareAndSet(false, true)) {
            try {
                // bounded wait for a late inference
                boolean acquired = runLock.tryAcquire(500, TimeUnit.MILLISECONDS);
                if (!acquired) {
                    if (errorListener != null) {
                        errorListener.onError("Interpreter close timed out waiting for late inference; leaking interpreter.", null);
                    }
                    return; // DO NOT close, leak it.
                }
                
                try {
                    adapter.close();
                } finally {
                    runLock.release();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
