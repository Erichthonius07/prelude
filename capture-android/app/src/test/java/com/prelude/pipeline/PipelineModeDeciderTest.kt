package com.prelude.pipeline

import com.prelude.capture.model.FrameBurst
import com.prelude.capture.model.FrameSample
import org.junit.Assert.assertEquals
import org.junit.Test

class PipelineModeDeciderTest {
    private val decider = PipelineModeDecider()

    @Test fun acceptedFramesUseMultiFrameFusion() {
        val burst = FrameBurst(4, listOf(FrameSample(0, .03, true), FrameSample(1, .03, true)))
        assertEquals("fusion_multi", decider.decide(burst).mode)
    }

    @Test fun allRejectedFramesUseEmergencySingleFrame() {
        val burst = FrameBurst(4, listOf(FrameSample(0, .01, false), FrameSample(1, .01, false)))
        assertEquals("fusion_single", decider.decide(burst).mode)
    }
}
