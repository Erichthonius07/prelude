package com.prelude.pipeline.contract

import android.graphics.Bitmap

/**
 * Stage-to-stage data contracts, in pipeline order:
 *
 *   CaptureModule    → FrameBurst          (com.prelude.capture.model)
 *   AlignmentModule  → AlignedFrameSet     + AlignmentMetrics
 *   PipelineModeDecider (alignment confidence only) → PipelineMode, decided before fusion runs
 *   FusionModule     → FusedFrame          + per-strategy metrics
 *   DenoiseModule    → DenoisedFrame       (discard-race timeout is internal to the module)
 *   PostProcessModule→ FinalImage
 *
 * The Results Service API contract is owned by Role 1. Until it is published, these types are the app-side
 * DRAFT; changes after Phase 2 must be add-only (new fields with defaults, no renames or removals).
 */
const val STAGE_CONTRACT_VERSION = "0.1-draft"

enum class PipelineMode(val wire: String) { FUSION_MULTI("fusion_multi"), FUSION_SINGLE("fusion_single") }

/**
 * One frame warped into the reference frame's geometry.
 *
 * [ransacInlierRatio] is RANSAC's actual post-alignment inlier ratio and is the ONLY confidence value fusion
 * may consume. The learned pre-alignment classifier's prediction deliberately does not appear on this type;
 * it lives in [AlignmentMetrics] for reporting only.
 */
class AlignedFrame(
    val sourceIndex: Int,
    val bitmap: Bitmap,
    /** 3x3 row-major homography mapping this frame onto the reference; identity for the reference itself. */
    val homography: FloatArray,
    val ransacInlierRatio: Double,
    /** False when RANSAC failed or the inlier ratio is under the device's alignment confidence floor. */
    val accepted: Boolean,
)

class AlignmentMetrics(
    val referenceIndex: Int,
    val attempted: Int,
    val succeeded: Int,
    val inlierRatios: List<Double>,
    /** Learned classifier's pre-alignment viability predictions, per attempted frame. Reporting only. */
    val classifierPredictions: List<Double?> = emptyList(),
    val latencyMs: Long,
    val note: String? = null,
)

class AlignedFrameSet(
    val burstId: String,
    val referenceIndex: Int,
    val frames: List<AlignedFrame>,
    val metrics: AlignmentMetrics,
) {
    val usableFrames: List<AlignedFrame> get() = frames.filter { it.accepted }
    val reference: AlignedFrame get() = frames.first { it.sourceIndex == referenceIndex }
}

enum class FusionStrategy(val wire: String, val label: String) {
    NAIVE_MEAN("naive_mean", "naive mean"),
    TRIMMED_MEAN("trimmed_mean", "trimmed mean"),
    CONFIDENCE_WEIGHTED("confidence_weighted", "weighted"),
    LEARNED("learned", "learned fusion"),
}

class FusionStrategyMetrics(
    val strategy: FusionStrategy,
    val available: Boolean,
    val latencyMs: Long,
    val framesUsed: Int,
    val note: String? = null,
)

class FusedFrame(
    val burstId: String,
    val bitmap: Bitmap,
    val mode: PipelineMode,
    /** Null in fusion_single: the reference frame is passed through. */
    val strategy: FusionStrategy?,
    val framesUsed: Int,
)

class DenoisedFrame(
    val burstId: String,
    val bitmap: Bitmap,
    /** False when no model ran (module not connected) or the discard-race was lost. */
    val applied: Boolean,
    val timedOut: Boolean,
    val latencyMs: Long,
    /** "nnapi", "cpu" or "none". */
    val backend: String,
    val modelId: String?,
)

class PostProcessParams(
    val toneGamma: Float = 0.8f,
    val contrast: Float = 1.08f,
    val sharpenAmount: Float = 0.6f,
    val sharpenRadius: Int = 1,
    /** Where these values came from, e.g. "grid-search 2026-10-20" or "default". */
    val source: String = "default",
)

class FinalImage(
    val burstId: String,
    val bitmap: Bitmap,
    val params: PostProcessParams,
    val latencyMs: Long,
)
