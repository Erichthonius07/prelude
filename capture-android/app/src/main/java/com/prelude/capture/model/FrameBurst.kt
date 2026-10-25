package com.prelude.capture.model

import android.graphics.Bitmap
import com.prelude.capture.calibration.TextureBucket
import java.time.Instant

/**
 * One captured frame plus its per-frame metadata (CaptureModule output contract).
 * Fields after [bitmap] were added add-only, with defaults, so older callers keep compiling.
 */
data class FrameSample(
    val index: Int,
    val varianceOfLaplacian: Double,
    val accepted: Boolean,
    val bitmap: Bitmap? = null,
    val timestampNs: Long? = null,
    val exposureTimeNs: Long? = null,
    val iso: Int? = null,
    val textureBucket: TextureBucket? = null,
    /** The per-texture-bucket blur threshold this frame was judged against. */
    val blurThreshold: Double? = null,
)

data class FrameBurst(
    val requestedCount: Int,
    val frames: List<FrameSample>,
    val capturedAt: Instant = Instant.now(),
    val burstId: String = "burst-${capturedAt.toEpochMilli()}",
) {
    val acceptedFrames: List<FrameSample> get() = frames.filter { it.accepted }
    val mode: String get() = if (acceptedFrames.size >= 2) "fusion_multi" else "fusion_single"

    /** Alignment reference: the sharpest accepted frame, not the first captured. */
    val referenceIndex: Int get() = (acceptedFrames.ifEmpty { frames }).maxByOrNull { it.varianceOfLaplacian }?.index ?: 0
}
