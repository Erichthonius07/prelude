package com.prelude.denoise.model;

import org.tensorflow.lite.InterpreterApi;
import com.prelude.denoise.timeout.ErrorListener;
import java.nio.ByteBuffer;

public class LiteRtAdapterImpl implements LiteRtAdapter {
    private final InterpreterApi interpreter;
    
    /** Thread count used for inference, documented for benchmark metadata inclusion. */
    public static final int NUM_THREADS = 4;

    public LiteRtAdapterImpl(ByteBuffer modelBuffer, ErrorListener errorListener) {
        this(modelBuffer, errorListener, NUM_THREADS);
    }

    /** Overload with explicit thread count; NUM_THREADS stays the documented default. */
    public LiteRtAdapterImpl(ByteBuffer modelBuffer, ErrorListener errorListener, int numThreads) {
        if (modelBuffer == null || !modelBuffer.isDirect()) {
            throw new IllegalArgumentException("Model buffer must be a direct ByteBuffer");
        }
        InterpreterApi.Options options = new InterpreterApi.Options();
        options.setUseXNNPACK(true);
        options.setNumThreads(numThreads);

        this.interpreter = InterpreterApi.create(modelBuffer, options);
        if (errorListener != null) {
            errorListener.onInfo("XNNPACK requested");
            errorListener.onInfo("numThreads=" + numThreads);
        }
    }
    
    @Override
    public void run(ByteBuffer in, ByteBuffer out) {
        interpreter.run(in, out);
    }
    
    @Override
    public void close() {
        interpreter.close();
    }
}
