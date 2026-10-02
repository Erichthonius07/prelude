package com.prelude.pipeline

/**
 * Pure decision: does a burst carry enough confident frames to run fusion over
 * multiple frames, or fall back to the single best frame?
 */
enum class PipelineMode { FUSION_MULTI, FUSION_SINGLE }

object PipelineModeDecider {
    /** Frames at or above [floor] survive; multi-fusion needs at least [minFrames] survivors. */
    fun decide(confidences: List<Float>, floor: Float, minFrames: Int): PipelineMode =
        if (confidences.count { it >= floor } >= minFrames) PipelineMode.FUSION_MULTI
        else PipelineMode.FUSION_SINGLE
}
