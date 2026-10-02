package com.prelude.denoise.model

/**
 * Input to the denoise module. Matches contract §6.3.
 *
 * @property burstId Becomes imageId for all downstream reporting (§2.3).
 * @property strategy Which fusion strategy produced this frame.
 * @property image Linear-domain fused frame pixel data.
 * @property pipelineMode fusion_multi | fusion_single (§6.6).
 *     MUST NOT influence the denoise timeout (§6.6, rules D4/T5).
 * @property evaluationSidecars Present in evaluation mode only (§6.3).
 */
data class FusedFrame(
    val burstId: String,
    val strategy: String,
    val image: FloatArray,
    val pipelineMode: String,
    val evaluationSidecars: Map<String, FloatArray>? = null,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is FusedFrame) return false
        return burstId == other.burstId &&
            strategy == other.strategy &&
            image.contentEquals(other.image) &&
            pipelineMode == other.pipelineMode
    }

    override fun hashCode(): Int {
        var result = burstId.hashCode()
        result = 31 * result + strategy.hashCode()
        result = 31 * result + image.contentHashCode()
        result = 31 * result + pipelineMode.hashCode()
        return result
    }
}
