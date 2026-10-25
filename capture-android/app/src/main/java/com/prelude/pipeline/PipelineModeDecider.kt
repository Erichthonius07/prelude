package com.prelude.pipeline

import com.prelude.capture.model.FrameBurst
import com.prelude.pipeline.contract.AlignedFrameSet
import com.prelude.pipeline.contract.PipelineMode

data class PipelineModeDecision(val mode: String, val reason: String) {
    val pipelineMode: PipelineMode get() = if (mode == PipelineMode.FUSION_MULTI.wire) PipelineMode.FUSION_MULTI else PipelineMode.FUSION_SINGLE
}

class PipelineModeDecider(private val minimumUsableFrames: Int = 2) {
    fun decide(burst: FrameBurst): PipelineModeDecision = if (burst.acceptedFrames.size >= minimumUsableFrames) {
        PipelineModeDecision("fusion_multi", "${burst.acceptedFrames.size} accepted frames survived blur rejection")
    } else {
        PipelineModeDecision("fusion_single", "all frames below blur threshold; using sharpest frame as emergency fallback")
    }

    /**
     * The orchestration flag: decided from the alignment confidence signal only (RANSAC accept/reject against
     * the confidence floor), before FusionModule runs. Does not govern the denoise timeout.
     */
    fun decide(aligned: AlignedFrameSet): PipelineModeDecision {
        val usable = aligned.usableFrames.size
        return if (usable >= minimumUsableFrames) {
            PipelineModeDecision(PipelineMode.FUSION_MULTI.wire, "$usable of ${aligned.frames.size} frames aligned above the confidence floor")
        } else {
            PipelineModeDecision(PipelineMode.FUSION_SINGLE.wire, aligned.metrics.note ?: "only $usable frame(s) aligned above the confidence floor; using the reference frame")
        }
    }
}
