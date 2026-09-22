package com.prelude.capture.calibration

import com.prelude.capture.capture.FrameFeatures

enum class ThresholdType { BLUR_TEXTURE, ALIGNMENT_CONFIDENCE_FLOOR, MIN_FRAME_COUNT, INFERENCE_TIMEOUT }

sealed class CalibrationOutcome {
    data class Success(val summary: String) : CalibrationOutcome()
    data class Blocked(val reason: String) : CalibrationOutcome()
}

/**
 * One deterministic calculator per threshold (spec: a single deterministic tool, not 3 off-device + 1 on).
 * Three of four are interface stubs pending other roles (§4 of the task brief).
 */
interface ThresholdCalculator {
    val type: ThresholdType
    /** Non-null when the calculator cannot run yet (blocked on another role). */
    val blockedReason: String?
    val isBlocked: Boolean get() = blockedReason != null

    /** Runs on-device; for blur/texture the input is a focus-sweep sample set. */
    suspend fun calibrate(samples: List<FrameFeatures>): CalibrationOutcome
}