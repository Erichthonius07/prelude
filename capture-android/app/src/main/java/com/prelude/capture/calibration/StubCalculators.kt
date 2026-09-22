package com.prelude.capture.calibration

import com.prelude.capture.capture.FrameFeatures

/** Blocked on Role 2 (RANSAC/homography must exist and be callable). */
class AlignmentConfidenceFloorCalculator : ThresholdCalculator {
    override val type = ThresholdType.ALIGNMENT_CONFIDENCE_FLOOR
    override val blockedReason =
        "Blocked: requires Role 2 RANSAC/homography implementation to produce confidence samples."
    override suspend fun calibrate(samples: List<FrameFeatures>) = CalibrationOutcome.Blocked(blockedReason!!)
}

/** Blocked on Role 2 (alignment) + Role 3 (naive fusion for the SSIM-gain stopping rule). */
class MinFrameCountCalculator : ThresholdCalculator {
    override val type = ThresholdType.MIN_FRAME_COUNT
    override val blockedReason =
        "Blocked: requires Role 2 alignment + Role 3 naive fusion (SSIM-gain stopping rule)."
    override suspend fun calibrate(samples: List<FrameFeatures>) = CalibrationOutcome.Blocked(blockedReason!!)
}

/** Blocked on Role 4 (trained model) + Role 5a (quantized TFLite). Must run REAL on-device inference. */
class InferenceTimeoutCalculator : ThresholdCalculator {
    override val type = ThresholdType.INFERENCE_TIMEOUT
    override val blockedReason =
        "Blocked: requires Role 4 trained model + Role 5a quantized TFLite; computes p95 + 20%, cap 500ms " +
        "from real on-device inference (a desktop script cannot measure this)."
    override suspend fun calibrate(samples: List<FrameFeatures>) = CalibrationOutcome.Blocked(blockedReason!!)
}