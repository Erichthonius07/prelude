package com.prelude.capture.model

/** One raw YUV_420_888 frame as delivered by Camera2 (plane bytes copied out). */
data class YuvFrame(
    val width: Int,
    val height: Int,
    val y: ByteArray,
    val u: ByteArray,
    val v: ByteArray,
)

/** A captured frame with the per-frame quality metadata submitted to the Results Service. */
data class CapturedFrame(
    val frameIndex: Int,
    // Field name per data-contract.md §6.1 ("image"). On-device type is YuvFrame —
    // Camera2 Image buffers are transient, so planes are copied out eagerly; ImageProxy
    // (CameraX) is not available to this Camera2-based module.
    val image: YuvFrame,
    val sensorTimestampNs: Long,
    val iso: Int,
    val exposureTimeNs: Long,
    val sharpnessScore: Float,
    val blurRejected: Boolean,
)

/** One burst: ordered frames plus the all-blurred emergency-fallback flag. */
data class FrameBurst(
    val burstId: String,
    val frames: List<CapturedFrame>,
    val emergencyFallbackActive: Boolean = false,
)
