package com.prelude.pipeline

import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import com.prelude.capture.calibration.DeviceThresholds
import com.prelude.capture.model.FrameBurst
import com.prelude.capture.net.MetricsOutbox
import com.prelude.pipeline.contract.AlignedFrameSet
import com.prelude.pipeline.contract.DenoisedFrame
import com.prelude.pipeline.contract.FinalImage
import com.prelude.pipeline.contract.FusionStrategy
import com.prelude.pipeline.contract.PipelineMode
import com.prelude.pipeline.fusion.FusionModule
import com.prelude.pipeline.modules.AlignmentModule
import com.prelude.pipeline.modules.DenoiseModule
import com.prelude.pipeline.modules.NotConnectedAlignment
import com.prelude.pipeline.modules.NotConnectedDenoise
import com.prelude.pipeline.modules.FusionWeightModel
import com.prelude.pipeline.postprocess.PostProcessModule
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import java.util.concurrent.Executors

class PipelineOutput(
    val burst: FrameBurst,
    val aligned: AlignedFrameSet,
    val decision: PipelineModeDecision,
    val fusion: FusionModule.Result,
    val denoised: DenoisedFrame,
    val final: FinalImage,
    /** Wall time from burst handed over to final image (excludes capture). */
    val processingMs: Long,
)

/**
 * Runs FrameBurst → align → pipeline_mode → fuse → denoise → post-process on a background thread, reports
 * progress on the main thread, and queues each stage's own metrics in the outbox.
 */
class PipelineRunner(
    private val alignment: AlignmentModule,
    private val fusion: FusionModule,
    private val denoise: DenoiseModule,
    val postProcess: PostProcessModule,
    private val decider: PipelineModeDecider,
    private val thresholds: DeviceThresholds,
    private val outbox: MetricsOutbox?,
    private val deviceId: String,
) {
    enum class Stage(val row: Int) { ALIGN(1), FUSE(2), DENOISE(3), POST(4) }

    interface Listener {
        fun onStageStarted(stage: Stage)
        fun onStageFinished(stage: Stage, summary: String, warn: Boolean)
        fun onAligned(aligned: AlignedFrameSet, decision: PipelineModeDecision)
        fun onPipelineDone(output: PipelineOutput)
        fun onPipelineFailed(stage: Stage, error: Throwable)
    }

    /** Best fusion strategy from the latest scored comparison; the live pipeline denoises this one. */
    @Volatile var preferredStrategy: FusionStrategy = FusionStrategy.CONFIDENCE_WEIGHTED

    private val executor = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())

    fun process(burst: FrameBurst, listener: Listener) {
        executor.execute {
            val start = SystemClock.elapsedRealtime()
            var stage = Stage.ALIGN
            fun started(s: Stage) { stage = s; main.post { listener.onStageStarted(s) } }
            fun finished(s: Stage, summary: String, warn: Boolean = false) { main.post { listener.onStageFinished(s, summary, warn) } }
            try {
                started(Stage.ALIGN)
                val aligned = alignment.align(burst, burst.referenceIndex, thresholds)
                val decision = decider.decide(aligned)
                main.post { listener.onAligned(aligned, decision) }
                val m = aligned.metrics
                if (m.note != null) finished(Stage.ALIGN, "${m.note}  ·  reference frame only", warn = true)
                else finished(Stage.ALIGN, "RANSAC %.2f mean  ·  %d of %d  ·  %d ms".format(Locale.US, m.inlierRatios.filter { !it.isNaN() }.average().takeIf { !it.isNaN() } ?: 0.0, aligned.usableFrames.size, aligned.frames.size, m.latencyMs))
                record(burst.burstId, "alignment", JSONObject()
                    .put("referenceIndex", m.referenceIndex).put("attempted", m.attempted).put("succeeded", m.succeeded)
                    .put("inlierRatios", JSONArray(m.inlierRatios.map { if (it.isNaN()) JSONObject.NULL else it }))
                    .put("latencyMs", m.latencyMs).put("note", m.note ?: JSONObject.NULL))
                record(burst.burstId, "pipeline_mode", JSONObject().put("mode", decision.mode).put("reason", decision.reason))

                started(Stage.FUSE)
                val fused = fusion.fuse(aligned, decision.pipelineMode, preferredStrategy)
                if (decision.pipelineMode == PipelineMode.FUSION_SINGLE) finished(Stage.FUSE, "skipped  ·  fusion_single, reference passed on", warn = true)
                else {
                    val ran = fused.metrics.filter { it.available }
                    finished(Stage.FUSE, "${fused.chosen.strategy?.label}  ·  ${fused.chosen.framesUsed} frames  ·  ${ran.size} of 4 strategies  ·  ${ran.sumOf { it.latencyMs }} ms", warn = ran.size < 4)
                }
                record(burst.burstId, "fusion", JSONObject()
                    .put("mode", decision.mode).put("chosen", fused.chosen.strategy?.wire ?: JSONObject.NULL)
                    .put("strategies", JSONArray(fused.metrics.map { s ->
                        JSONObject().put("strategy", s.strategy.wire).put("available", s.available).put("latencyMs", s.latencyMs).put("framesUsed", s.framesUsed).put("note", s.note ?: JSONObject.NULL)
                    })))

                started(Stage.DENOISE)
                val denoised = denoise.denoise(fused.chosen, thresholds.inferenceTimeoutMs)
                when {
                    denoised.timedOut -> finished(Stage.DENOISE, "lost discard-race at ${thresholds.inferenceTimeoutMs} ms  ·  fused frame kept", warn = true)
                    !denoised.applied -> finished(Stage.DENOISE, "not connected  ·  fused frame passed on", warn = true)
                    else -> finished(Stage.DENOISE, "${denoised.backend}  ·  ${denoised.modelId ?: "model"}  ·  ${denoised.latencyMs} ms")
                }
                record(burst.burstId, "denoise", JSONObject()
                    .put("applied", denoised.applied).put("timedOut", denoised.timedOut).put("latencyMs", denoised.latencyMs)
                    .put("timeoutMs", thresholds.inferenceTimeoutMs).put("backend", denoised.backend).put("modelId", denoised.modelId ?: JSONObject.NULL))

                started(Stage.POST)
                val final = postProcess.process(denoised)
                finished(Stage.POST, "tone map + sharpen  ·  ${final.params.source}  ·  ${final.latencyMs} ms")
                record(burst.burstId, "postprocess", JSONObject()
                    .put("latencyMs", final.latencyMs).put("toneGamma", final.params.toneGamma.toDouble()).put("contrast", final.params.contrast.toDouble())
                    .put("sharpenAmount", final.params.sharpenAmount.toDouble()).put("sharpenRadius", final.params.sharpenRadius).put("paramsSource", final.params.source))

                val output = PipelineOutput(burst, aligned, decision, fused, denoised, final, SystemClock.elapsedRealtime() - start)
                main.post { listener.onPipelineDone(output) }
            } catch (t: Throwable) {
                val failed = stage
                main.post { listener.onPipelineFailed(failed, t) }
            }
        }
    }

    private fun record(burstId: String, stage: String, metrics: JSONObject) {
        runCatching { outbox?.record(deviceId, burstId, stage, metrics) }
    }

    fun shutdown() { executor.shutdown() }
}

/**
 * The one place modules are wired together. To connect a teammate's module, replace the NotConnected* instance:
 *  - Role 2 alignment:      alignment = YourOpenCvAlignment(...)
 *  - Role 5a denoiser:      denoise = YourTfliteDenoiser(context, "denoiser_int8.tflite")
 *  - Role 3 learned fusion: learnedFusion = YourFusionWeightModel(context, "fusion_weights.tflite")
 */
object PipelineFactory {
    fun create(thresholds: DeviceThresholds, outbox: MetricsOutbox?, deviceId: String): PipelineRunner {
        val alignment: AlignmentModule = NotConnectedAlignment()
        val denoise: DenoiseModule = NotConnectedDenoise()
        val learnedFusion: FusionWeightModel? = null
        return PipelineRunner(alignment, FusionModule(learnedFusion), denoise, PostProcessModule(), PipelineModeDecider(), thresholds, outbox, deviceId)
    }
}
