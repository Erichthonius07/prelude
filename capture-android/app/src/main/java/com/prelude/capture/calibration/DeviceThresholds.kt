package com.prelude.capture.calibration

data class BlurTextureThreshold(
    val bucketEdges: List<Float>,       // texture boundaries (size = buckets-1)
    val bucketSharpnessMin: List<Float> // sharpness floor per bucket (size = buckets)
) {
    fun isSharp(texture: Float, sharpness: Float): Boolean {
        var b = 0
        while (b < bucketEdges.size && texture > bucketEdges[b]) b++
        val floor = bucketSharpnessMin.getOrElse(b) { 0f }
        return sharpness >= floor
    }
}

data class DeviceThresholds(
    val blurTexture: BlurTextureThreshold? = null,   // computed on-device (Role 1)
    val alignmentConfidenceFloor: Float? = null,      // null until Role 2
    val minFrameCount: Int? = null,                   // null until Roles 2+3
    val inferenceTimeoutMs: Long? = null,             // null until Roles 4+5a
)