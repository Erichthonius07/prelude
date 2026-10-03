package com.prelude.denoise.model;

import java.nio.ByteBuffer;

public interface LiteRtAdapter extends AutoCloseable {
    void run(ByteBuffer in, ByteBuffer out);
    @Override
    void close();
}
