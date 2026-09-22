package com.prelude.capture

import android.content.Context
import android.os.Build
import android.util.Log
import com.prelude.capture.calibration.CalibrationRepository
import com.prelude.capture.capture.BurstCaptureManager
import com.prelude.capture.capture.SharpnessEstimator
import com.prelude.capture.model.CapturedFrame
import com.prelude.capture.model.FrameBurst
import com.prelude.capture.net.ResultsServiceClient

class CaptureModule(
    private val context: Context,
    private val resultsBaseUrl: String,
    private val runId: String, // FLAG: live captures need a pre-registered runId — see open questions.
) {
    private val repo = CalibrationRepository(context)
    private val client = ResultsServiceClient(resultsBaseUrl)
    private val capture = BurstCaptureManager(context)

    companion object { const val MIN_FRAMES = 4; const val MAX_FRAMES = 8 }

    suspend fun captureAndSubmit(numFrames: Int = MAX_FRAMES): FrameBurst {
        val thresholds = repo.load()
        val blur = thresholds.blurTexture
        capture.open()

        val raw = capture.captureBurst(numFrames.coerceIn(MIN_FRAMES, MAX_FRAMES))
        val frames = raw.mapIndexed { i, (yuv, result) ->
            val (iso, exp, ts) = capture.metadata(result)
            val sharpness = SharpnessEstimator.varianceOfLaplacian(yuv)
            val rejected = blur?.isSharp(SharpnessEstimator.textureEnergy(yuv), sharpness)?.not() ?: false
            CapturedFrame(i, yuv, ts, iso, exp, sharpness, rejected)
        }

        // Emergency fallback: if every frame is blur-rejected, keep the single highest-sharpness
        // frame and flag the all-blurred event.
        val allBlurred = frames.isNotEmpty() && frames.all { it.blurRejected }
        val finalFrames = if (allBlurred) {
            val best = frames.maxByOrNull { it.sharpnessScore }!!
            frames.map { if (it === best) it.copy(blurRejected = false) else it }
        } else frames

        val burst = FrameBurst(
            burstId = "${Build.MODEL}:${System.nanoTime()}",
            frames = finalFrames,
            emergencyFallbackActive = allBlurred,
        )
        if (allBlurred) Log.w("CaptureModule", "ALL-BLURRED burst ${burst.burstId}; fallback to best frame")

        client.submitCapture(runId, Build.MODEL, "0.1.0", listOf(burst))
        capture.close()
        return burst
    }
}