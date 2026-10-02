package com.prelude.denoise.model

/**
 * Output of the denoise module. Matches contract §6.4.
 *
 * On-device field names differ from wire names (rule D2):
 * - `latencyNs` (Long) → `latencyMs` (Double): `latencyNs / 1_000_000.0`
 * - `discardRaceOccurred` → `discardRaceEvent`: rename only
 * - `timeoutOccurred` → `timeoutEvent`: rename only
 *
 * @property burstId Same as input burstId, carried unchanged (rule D3).
 * @property image Denoised pixel data, or fusion-only pixels if timeout.
 * @property modelVariant e.g. "cnn-v1" (§5.4/§6.4, rule C6).
 * @property precision "fp32" or "int8".
 * @property latencyNs Inference latency in nanoseconds; 0 if timeout.
 * @property discardRaceOccurred True only when timer won AND late inference
 *     result was dropped (rule T3).
 * @property timeoutOccurred True when the timer won the race (rule T3).
 */
data class DenoisedFrame(
    val burstId: String,
    val image: FloatArray,
    val modelVariant: String,
    val precision: String,
    val latencyNs: Long,
    val discardRaceOccurred: Boolean,
    val timeoutOccurred: Boolean,
) {
    /** Wire-format latency in milliseconds (§5.4). */
    val latencyMs: Double get() = latencyNs / 1_000_000.0

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DenoisedFrame) return false
        return burstId == other.burstId &&
            image.contentEquals(other.image) &&
            modelVariant == other.modelVariant &&
            precision == other.precision &&
            latencyNs == other.latencyNs &&
            discardRaceOccurred == other.discardRaceOccurred &&
            timeoutOccurred == other.timeoutOccurred
    }

    override fun hashCode(): Int {
        var result = burstId.hashCode()
        result = 31 * result + image.contentHashCode()
        result = 31 * result + modelVariant.hashCode()
        result = 31 * result + precision.hashCode()
        result = 31 * result + latencyNs.hashCode()
        result = 31 * result + discardRaceOccurred.hashCode()
        result = 31 * result + timeoutOccurred.hashCode()
        return result
    }
}
