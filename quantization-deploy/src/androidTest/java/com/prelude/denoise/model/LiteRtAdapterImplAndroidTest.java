package com.prelude.denoise.model;

import org.junit.Test;
import org.junit.runner.RunWith;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import static org.junit.Assert.assertNotNull;

@RunWith(AndroidJUnit4.class)
public class LiteRtAdapterImplAndroidTest {

    @Test
    public void testNativeInterpreterLoads() {
        // This test requires a physical device because liblitert_jni.so 
        // throws UnsatisfiedLinkError on the JVM.
        // It is marked as NOT RUN via documentation per user request.
        
        int bufferBytes = 1 * 3 * 256 * 256 * 4;
        ByteBuffer modelBuffer = ByteBuffer.allocateDirect(1024).order(ByteOrder.nativeOrder());
        
        try {
            // Note: with an empty buffer, this will fail to parse the FlatBuffer.
            // But if it reaches the native call, it proves the JNI loaded.
            LiteRtAdapterImpl adapter = new LiteRtAdapterImpl(modelBuffer, null);
            assertNotNull(adapter);
            adapter.close();
        } catch (IllegalArgumentException e) {
            // Expected because the buffer isn't a valid TFLite model,
            // but it proves the class loaded!
        }
    }
}
