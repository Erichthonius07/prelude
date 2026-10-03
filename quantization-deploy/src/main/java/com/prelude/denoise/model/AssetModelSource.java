package com.prelude.denoise.model;

import android.content.res.AssetManager;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * ModelSource that loads a .tflite model from the APK assets (decision A24:
 * the single final small INT8 model is bundled as an asset).
 */
public class AssetModelSource implements ModelSource {

    private final AssetManager assetManager;
    private final String assetPath;

    public AssetModelSource(AssetManager assetManager, String assetPath) {
        if (assetManager == null || assetPath == null || assetPath.isEmpty()) {
            throw new IllegalArgumentException("assetManager and assetPath must be non-empty");
        }
        this.assetManager = assetManager;
        this.assetPath = assetPath;
    }

    @Override
    public ByteBuffer loadModel() {
        try (InputStream is = assetManager.open(assetPath)) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] chunk = new byte[8192];
            int read;
            while ((read = is.read(chunk)) > 0) {
                out.write(chunk, 0, read);
            }
            ByteBuffer buffer = ByteBuffer.allocateDirect(out.size()).order(ByteOrder.nativeOrder());
            buffer.put(out.toByteArray());
            buffer.flip();
            return buffer;
        } catch (IOException e) {
            throw new RuntimeException("Failed to load model from assets: " + assetPath, e);
        }
    }
}
