package com.prelude.pipeline

import org.junit.Assert.assertEquals
import org.junit.Test

class PipelineModeDeciderTest {

    // --- normal cases ---

    @Test
    fun `enough surviving frames gives multi`() {
        val conf = listOf(0.9f, 0.8f, 0.85f, 0.7f) // 4 survive
        assertEquals(PipelineMode.FUSION_MULTI, PipelineModeDecider.decide(conf, 0.5f, 3))
    }

    @Test
    fun `surviving exactly equals minimum gives multi`() {
        val conf = listOf(0.9f, 0.8f, 0.7f, 0.2f) // 3 survive, min = 3 -> not below
        assertEquals(PipelineMode.FUSION_MULTI, PipelineModeDecider.decide(conf, 0.5f, 3))
    }

    @Test
    fun `surviving one below minimum gives single`() {
        val conf = listOf(0.9f, 0.8f, 0.2f, 0.1f) // 2 survive, min = 3 -> below
        assertEquals(PipelineMode.FUSION_SINGLE, PipelineModeDecider.decide(conf, 0.5f, 3))
    }

    // --- exactly at the threshold ---

    @Test
    fun `confidence exactly at floor survives`() {
        val conf = listOf(0.5f, 0.5f, 0.5f)
        assertEquals(PipelineMode.FUSION_MULTI, PipelineModeDecider.decide(conf, 0.5f, 3))
    }

    @Test
    fun `confidence just below floor does not survive`() {
        val conf = listOf(0.49f, 0.49f, 0.49f)
        assertEquals(PipelineMode.FUSION_SINGLE, PipelineModeDecider.decide(conf, 0.5f, 3))
    }

    // --- empty list ---

    @Test
    fun `empty burst gives single`() {
        assertEquals(PipelineMode.FUSION_SINGLE, PipelineModeDecider.decide(emptyList(), 0.5f, 3))
    }

    @Test
    fun `empty burst with minimum one gives single`() {
        assertEquals(PipelineMode.FUSION_SINGLE, PipelineModeDecider.decide(emptyList(), 0.5f, 1))
    }

    // --- single frame ---

    @Test
    fun `single surviving frame with minimum one gives multi`() {
        assertEquals(PipelineMode.FUSION_MULTI, PipelineModeDecider.decide(listOf(0.9f), 0.5f, 1))
    }

    @Test
    fun `single surviving frame with minimum two gives single`() {
        assertEquals(PipelineMode.FUSION_SINGLE, PipelineModeDecider.decide(listOf(0.9f), 0.5f, 2))
    }

    @Test
    fun `single frame below floor gives single`() {
        assertEquals(PipelineMode.FUSION_SINGLE, PipelineModeDecider.decide(listOf(0.3f), 0.5f, 1))
    }

    // --- all below floor ---

    @Test
    fun `all frames below floor gives single`() {
        val conf = listOf(0.1f, 0.2f, 0.3f, 0.4f)
        assertEquals(PipelineMode.FUSION_SINGLE, PipelineModeDecider.decide(conf, 0.5f, 2))
    }
}