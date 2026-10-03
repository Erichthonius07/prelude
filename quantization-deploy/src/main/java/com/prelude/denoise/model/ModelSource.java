package com.prelude.denoise.model;

import java.nio.ByteBuffer;

public interface ModelSource {
    ByteBuffer loadModel();
}
