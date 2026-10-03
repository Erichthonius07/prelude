package com.prelude.denoise.model;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class FileModelSource implements ModelSource {
    private final String filePath;

    public FileModelSource(String filePath) {
        this.filePath = filePath;
    }

    @Override
    public ByteBuffer loadModel() {
        try (FileInputStream fis = new FileInputStream(new File(filePath));
             FileChannel channel = fis.getChannel()) {
            ByteBuffer buffer = ByteBuffer.allocateDirect((int) channel.size());
            buffer.order(ByteOrder.nativeOrder());
            channel.read(buffer);
            buffer.flip();
            return buffer;
        } catch (IOException e) {
            throw new RuntimeException("Failed to load model from file: " + filePath, e);
        }
    }
}
