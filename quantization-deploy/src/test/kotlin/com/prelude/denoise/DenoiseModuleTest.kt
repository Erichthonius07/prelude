package com.prelude.denoise

import com.prelude.denoise.model.FusedFrame
import com.prelude.denoise.timeout.FakeInferenceRunner
import com.prelude.denoise.timeout.FakeTimeSource
import com.prelude.denoise.timeout.InferenceRunner
import com.prelude.denoise.timeout.RaceResult
import com.prelude.denoise.timeout.SystemTimeSource
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * Tests for [DenoiseModule]: single-flight policy (T6/A26),
 * inference thread survival (T1), and pipelineMode isolation (T5).
 */
class DenoiseModuleTest {

    private fun input(burstId: String = "burst-1") = FusedFrame(
        burstId = burstId,
        strategy = "naive",
        image = floatArrayOf(0.1f, 0.2f, 0.3f),
        pipelineMode = "fusion_multi",
    )

    // ---- T7 test 7: Repeated bursts / single-flight / BUSY ----

    @Test
    fun singleFlightSecondBurstWhileInferenceIsLateGetsBusyFusionOnly() {
        val inferStarted = CountDownLatch(1)
        val inferProceed = CountDownLatch(1)
        val fakeInference = FakeInferenceRunner(
            result = floatArrayOf(0.9f),
            startedLatch = inferStarted,
            proceedLatch = inferProceed,
        )
        val fakeTime = FakeTimeSource()
        val module = DenoiseModule(fakeInference, fakeTime, "cnn-v1", "fp32")

        // First burst — will timeout (inference blocked on proceedLatch)
        val firstRef = AtomicReference<RaceResult>()
        val firstDone = CountDownLatch(1)
        Thread {
            firstRef.set(module.denoise(input("burst-1"), 100, DenoiseModule.RunMode.LIVE))
            firstDone.countDown()
        }.start()

        assertTrue("Inference should start", inferStarted.await(5, TimeUnit.SECONDS))
        fakeTime.releaseSleep() // timer wins; inference still blocked
        assertTrue("First denoise should return", firstDone.await(5, TimeUnit.SECONDS))
        assertTrue("First: timer won", firstRef.get()!!.timerWon)

        // Second burst — should get BUSY (inference from burst-1 still running)
        val secondResult = module.denoise(input("burst-2"), 100, DenoiseModule.RunMode.LIVE)

        assertTrue("Second: timeout (BUSY)", secondResult.frame.timeoutOccurred)
        assertEquals("Second: burstId preserved (D3)", "burst-2", secondResult.frame.burstId)
        assertArrayEquals(
            "BUSY frame should have fusion-only pixels",
            input("burst-2").image, secondResult.frame.image, 0.0001f,
        )

        // Release first inference — should clear busy flag
        inferProceed.countDown()
        firstRef.get()!!.awaitLateInference(5000)
    }

    // ---- T7 test 9: Inference thread survives after timer wins (T1) ----

    @Test
    fun inferenceThreadContinuesRunningAfterTimerWins() {
        val inferStarted = CountDownLatch(1)
        val inferProceed = CountDownLatch(1)
        val inferCompleted = CountDownLatch(1)

        val fakeInference = object : InferenceRunner {
            override fun run(input: FusedFrame): FloatArray {
                inferStarted.countDown()
                inferProceed.await()
                inferCompleted.countDown()
                return floatArrayOf(0.9f)
            }
        }
        val fakeTime = FakeTimeSource()
        val module = DenoiseModule(fakeInference, fakeTime, "cnn-v1", "fp32")

        val resultRef = AtomicReference<RaceResult>()
        val done = CountDownLatch(1)
        Thread {
            resultRef.set(module.denoise(input(), 50, DenoiseModule.RunMode.LIVE))
            done.countDown()
        }.start()

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS))
        fakeTime.releaseSleep() // timer wins
        assertTrue(done.await(5, TimeUnit.SECONDS))

        // run() returned, but inference thread is still alive (T1: never killed)
        val lateThread = resultRef.get()!!.lateInferenceThread
        assertNotNull("Late inference thread should exist", lateThread)
        assertTrue("Inference thread should still be alive", lateThread!!.isAlive)

        // Release inference — it should complete naturally
        inferProceed.countDown()
        assertTrue(
            "Inference thread should complete (not killed — T1)",
            inferCompleted.await(5, TimeUnit.SECONDS),
        )
    }

    // ---- T5: pipelineMode does not affect timeout ----

    @Test
    fun pipelineModeDoesNotAffectTimeoutBehavior() {
        // Run with fusion_multi — inference wins
        val runner1 = FakeInferenceRunner(result = floatArrayOf(0.5f))
        val module1 = DenoiseModule(runner1, SystemTimeSource, "cnn-v1", "fp32")
        val result1 = module1.denoise(
            input().copy(pipelineMode = "fusion_multi"), 500, DenoiseModule.RunMode.LIVE,
        )

        // Run with fusion_single — inference also wins
        val runner2 = FakeInferenceRunner(result = floatArrayOf(0.5f))
        val module2 = DenoiseModule(runner2, SystemTimeSource, "cnn-v1", "fp32")
        val result2 = module2.denoise(
            input().copy(pipelineMode = "fusion_single"), 500, DenoiseModule.RunMode.LIVE,
        )

        // Both should produce the same outcome regardless of pipelineMode
        assertEquals(
            "pipelineMode must not affect timeout (T5)",
            result1.inferenceWon, result2.inferenceWon,
        )
    }

    // ---- Timeout bounds ----

    @Test(expected = IllegalArgumentException::class)
    fun rejectsTimeoutAboveCeiling() {
        val module = DenoiseModule(FakeInferenceRunner(), modelVariant = "cnn-v1")
        module.denoise(input(), 501, DenoiseModule.RunMode.LIVE)
    }
}
