package com.prelude.denoise.timeout

import com.prelude.denoise.model.FusedFrame
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * Tests for [DiscardRaceRunner] covering T7 requirements.
 * Uses injectable clock ([FakeTimeSource]) and fake inferencer
 * ([FakeInferenceRunner]) for deterministic results.
 */
class DiscardRaceRunnerTest {

    private val testInput = FusedFrame(
        burstId = "test-burst-1",
        strategy = "naive",
        image = floatArrayOf(0.1f, 0.2f, 0.3f),
        pipelineMode = "fusion_multi",
    )

    // ---- T7 test 1: Timer wins (slow inference) ----

    @Test
    fun timerWinsWhenInferenceIsSlow() {
        val inferStarted = CountDownLatch(1)
        val inferProceed = CountDownLatch(1)
        val fakeInference = FakeInferenceRunner(
            result = floatArrayOf(0.9f),
            startedLatch = inferStarted,
            proceedLatch = inferProceed,
        )
        val fakeTime = FakeTimeSource()
        val runner = DiscardRaceRunner(fakeInference, fakeTime)

        val resultRef = AtomicReference<RaceResult>()
        val errorRef = AtomicReference<Throwable>()
        val done = CountDownLatch(1)

        Thread {
            try {
                resultRef.set(runner.run(testInput, 100, "cnn-v1", "fp32"))
            } catch (e: Throwable) {
                errorRef.set(e)
            }
            done.countDown()
        }.start()

        assertTrue("Inference should start", inferStarted.await(5, TimeUnit.SECONDS))
        fakeTime.releaseSleep() // timer fires; inference still blocked
        assertTrue("run() should complete", done.await(5, TimeUnit.SECONDS))
        assertNull("No error expected", errorRef.get())

        val result = resultRef.get()!!
        assertTrue("Timer should have won", result.timerWon)
        assertFalse("Inference should not have won", result.inferenceWon)
        assertTrue("Frame should indicate timeout", result.frame.timeoutOccurred)
        assertFalse("Discard not yet (inference still running)", result.discardRaceOccurred)
        assertArrayEquals("Fusion-only pixels", testInput.image, result.frame.image, 0.0001f)

        // Release inference — should trigger DISCARDED transition
        inferProceed.countDown()
        val discarded = result.awaitLateInference(5000)
        assertTrue("Late inference should be discarded", discarded)
        assertTrue("discardRaceOccurred should now be true", result.discardRaceOccurred)
    }

    // ---- T7 test 2: Inference wins (fast inference) ----

    @Test
    fun inferenceWinsWhenFasterThanTimer() {
        val fakeInference = FakeInferenceRunner(
            result = floatArrayOf(0.9f),
            delayMs = 0, // instant
        )
        // Use SystemTimeSource — inference finishes before timer can fire
        val runner = DiscardRaceRunner(fakeInference, SystemTimeSource)

        val result = runner.run(testInput, 500, "cnn-v1", "fp32")

        assertTrue("Inference should have won", result.inferenceWon)
        assertFalse("Timer should not have won", result.timerWon)
        assertFalse("No timeout", result.frame.timeoutOccurred)
        assertFalse("No discard", result.frame.discardRaceOccurred)
        assertArrayEquals("Denoised pixels returned", floatArrayOf(0.9f), result.frame.image, 0.0001f)
        assertTrue("Latency should be recorded", result.frame.latencyNs >= 0)
    }

    // ---- T7 test 3: Near-simultaneous ----

    @Test
    fun nearSimultaneousExactlyOneWinner() {
        val fakeInference = FakeInferenceRunner(
            result = floatArrayOf(0.7f),
            delayMs = 5,
        )
        val runner = DiscardRaceRunner(fakeInference, SystemTimeSource)

        // Timeout very close to inference time — either could win
        val result = runner.run(testInput, 8, "cnn-v1", "fp32")

        assertTrue(
            "Exactly one winner",
            result.inferenceWon xor result.timerWon,
        )
        assertNotNull("Frame always present", result.frame)
        assertEquals("BurstId preserved", "test-burst-1", result.frame.burstId)
    }

    // ---- T7 test 4: Late result never delivered ----

    @Test
    fun lateInferenceResultIsNeverDelivered() {
        val inferStarted = CountDownLatch(1)
        val inferProceed = CountDownLatch(1)
        val fakeInference = FakeInferenceRunner(
            result = floatArrayOf(0.9f), // different from fusion-only
            startedLatch = inferStarted,
            proceedLatch = inferProceed,
        )
        val fakeTime = FakeTimeSource()
        val runner = DiscardRaceRunner(fakeInference, fakeTime)

        val resultRef = AtomicReference<RaceResult>()
        val done = CountDownLatch(1)
        Thread {
            resultRef.set(runner.run(testInput, 50, "cnn-v1", "fp32"))
            done.countDown()
        }.start()

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS))
        fakeTime.releaseSleep()
        assertTrue(done.await(5, TimeUnit.SECONDS))

        val result = resultRef.get()!!
        assertArrayEquals(
            "Delivered pixels should be fusion-only",
            testInput.image, result.frame.image, 0.0001f,
        )

        // Let late inference finish
        inferProceed.countDown()
        result.awaitLateInference(5000)

        // Delivered frame is STILL fusion-only (T3: late result discarded)
        assertArrayEquals(
            "Delivered pixels must not change after late inference",
            testInput.image, result.frame.image, 0.0001f,
        )
        assertTrue("discardRaceOccurred should be true", result.discardRaceOccurred)
    }

    // ---- T7 test 5: No double emission ----

    @Test
    fun exactlyOneResultDeliveredNoDoubleEmission() {
        val deliveryCount = AtomicInteger(0)
        val inferStarted = CountDownLatch(1)
        val inferProceed = CountDownLatch(1)
        val fakeInference = FakeInferenceRunner(
            result = floatArrayOf(0.9f),
            startedLatch = inferStarted,
            proceedLatch = inferProceed,
        )
        val fakeTime = FakeTimeSource()
        val runner = DiscardRaceRunner(fakeInference, fakeTime)

        val resultRef = AtomicReference<RaceResult>()
        val done = CountDownLatch(1)
        Thread {
            resultRef.set(runner.run(testInput, 50, "cnn-v1", "fp32"))
            deliveryCount.incrementAndGet()
            done.countDown()
        }.start()

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS))
        fakeTime.releaseSleep()
        assertTrue(done.await(5, TimeUnit.SECONDS))
        assertEquals("Exactly one delivery", 1, deliveryCount.get())

        // Let late inference finish — must NOT cause another delivery
        inferProceed.countDown()
        resultRef.get()!!.awaitLateInference(5000)
        assertEquals("Still exactly one delivery", 1, deliveryCount.get())
    }

    // ---- T7 test 6: Interpreter never called concurrently ----

    @Test
    fun inferenceRunnerNeverCalledConcurrently() {
        val fakeInference = FakeInferenceRunner(
            result = floatArrayOf(0.9f),
            delayMs = 50,
        )
        val runner = DiscardRaceRunner(fakeInference, SystemTimeSource)

        runner.run(testInput, 200, "cnn-v1", "fp32")

        assertEquals(
            "Max concurrent calls should be 1 (D4)",
            1, fakeInference.maxConcurrentCalls.get(),
        )
    }

    // ---- T7 test 8a: Exception propagates when inference wins the race ----

    @Test
    fun inferenceExceptionPropagatesWhenInferenceWinsRace() {
        val fakeInference = FakeInferenceRunner(
            shouldThrow = RuntimeException("OOM in native"),
            delayMs = 0, // fails immediately → wins race
        )
        val fakeTime = FakeTimeSource()
        val runner = DiscardRaceRunner(fakeInference, fakeTime)

        val errorRef = AtomicReference<Throwable>()
        val done = CountDownLatch(1)

        Thread {
            try {
                runner.run(testInput, 100, "cnn-v1", "fp32")
            } catch (e: Throwable) {
                errorRef.set(e)
            }
            done.countDown()
        }.start()

        assertTrue(done.await(5, TimeUnit.SECONDS))
        assertNotNull("Exception should propagate", errorRef.get())
        assertEquals("OOM in native", errorRef.get()!!.message)

        // Clean up: release the timer thread blocked in FakeTimeSource
        try {
            fakeTime.releaseSleep()
        } catch (_: Exception) {
            // Timer thread may have already exited
        }
    }

    // ---- T7 test 8b: Exception swallowed when timer already won ----

    @Test
    fun inferenceExceptionSwallowedWhenTimerAlreadyWon() {
        val inferStarted = CountDownLatch(1)
        val inferProceed = CountDownLatch(1)
        val fakeInference = FakeInferenceRunner(
            shouldThrow = RuntimeException("OOM in native"),
            startedLatch = inferStarted,
            proceedLatch = inferProceed,
        )
        val fakeTime = FakeTimeSource()
        val lateCompleteCalled = CountDownLatch(1)
        val runner = DiscardRaceRunner(fakeInference, fakeTime)

        val resultRef = AtomicReference<RaceResult>()
        val done = CountDownLatch(1)
        Thread {
            resultRef.set(runner.run(testInput, 50, "cnn-v1", "fp32") {
                lateCompleteCalled.countDown()
            })
            done.countDown()
        }.start()

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS))
        fakeTime.releaseSleep() // timer wins
        assertTrue(done.await(5, TimeUnit.SECONDS))

        val result = resultRef.get()!!
        assertTrue("Timer should have won", result.timerWon)

        // Release inference — it will throw, but exception is swallowed
        inferProceed.countDown()
        assertTrue(
            "onLateInferenceComplete should be called even on error",
            lateCompleteCalled.await(5, TimeUnit.SECONDS),
        )
        // Failed late inference is not a discard (no result to drop)
        assertFalse("Failed late inference is not a discard", result.discardRaceOccurred)
    }

    // ---- onLateInferenceComplete callback fires after discard ----

    @Test
    fun onLateInferenceCompleteCallbackFiresAfterDiscard() {
        val inferStarted = CountDownLatch(1)
        val inferProceed = CountDownLatch(1)
        val callbackFired = CountDownLatch(1)
        val fakeInference = FakeInferenceRunner(
            result = floatArrayOf(0.9f),
            startedLatch = inferStarted,
            proceedLatch = inferProceed,
        )
        val fakeTime = FakeTimeSource()
        val runner = DiscardRaceRunner(fakeInference, fakeTime)

        val resultRef = AtomicReference<RaceResult>()
        val done = CountDownLatch(1)
        Thread {
            resultRef.set(runner.run(testInput, 50, "cnn-v1", "fp32") {
                callbackFired.countDown()
            })
            done.countDown()
        }.start()

        assertTrue(inferStarted.await(5, TimeUnit.SECONDS))
        fakeTime.releaseSleep()
        assertTrue(done.await(5, TimeUnit.SECONDS))

        assertEquals("Callback not yet", 1L, callbackFired.count)

        inferProceed.countDown()
        assertTrue(
            "Callback should fire after late inference completes",
            callbackFired.await(5, TimeUnit.SECONDS),
        )
    }

    // ---- Timeout bounds (T4) ----

    @Test(expected = IllegalArgumentException::class)
    fun rejectsTimeoutAbove500ms() {
        val runner = DiscardRaceRunner(FakeInferenceRunner(), SystemTimeSource)
        runner.run(testInput, 501, "cnn-v1", "fp32")
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsTimeoutOfZero() {
        val runner = DiscardRaceRunner(FakeInferenceRunner(), SystemTimeSource)
        runner.run(testInput, 0, "cnn-v1", "fp32")
    }

    // ---- Contract field validation ----

    @Test
    fun outputCarriesBurstIdUnchanged() {
        val fakeInference = FakeInferenceRunner(result = floatArrayOf(0.5f))
        val runner = DiscardRaceRunner(fakeInference, SystemTimeSource)
        val result = runner.run(testInput, 500, "cnn-v1", "fp32")
        assertEquals("burstId must be carried unchanged (D3)", "test-burst-1", result.frame.burstId)
    }

    @Test
    fun outputCarriesModelVariantAndPrecision() {
        val fakeInference = FakeInferenceRunner(result = floatArrayOf(0.5f))
        val runner = DiscardRaceRunner(fakeInference, SystemTimeSource)
        val result = runner.run(testInput, 500, "cnn-v1", "int8")
        assertEquals("cnn-v1", result.frame.modelVariant)
        assertEquals("int8", result.frame.precision)
    }

    @Test
    fun latencyMsConversionIsCorrect() {
        val fakeInference = FakeInferenceRunner(result = floatArrayOf(0.5f))
        val runner = DiscardRaceRunner(fakeInference, SystemTimeSource)
        val result = runner.run(testInput, 500, "cnn-v1", "fp32")
        assertEquals(
            "latencyMs = latencyNs / 1_000_000.0 (D2)",
            result.frame.latencyNs / 1_000_000.0,
            result.frame.latencyMs,
            0.001,
        )
    }
}
