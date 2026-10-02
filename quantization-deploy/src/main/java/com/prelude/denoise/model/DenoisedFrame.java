package com.prelude.denoise.model;

import java.util.Arrays;

/**
 * Output of the denoise module. Matches contract §6.4.
 *
 * On-device field names differ from wire names (rule D2):
 *   latencyNs (long) → latencyMs (double): latencyNs / 1_000_000.0
 *   discardRaceOccurred → discardRaceEvent: rename only
 *   timeoutOccurred → timeoutEvent: rename only
 *
 * burstId              — same as input burstId, carried unchanged (rule D3).
 * image                — denoised pixel data, or fusion-only pixels if timeout.
 * modelVariant         — e.g. "cnn-v1" (§5.4/§6.4, rule C6).
 * precision            — "fp32" or "int8".
 * latencyNs            — inference latency in nanoseconds; 0 if timeout.
 * discardRaceOccurred  — true only when timer won AND late inference result was dropped (rule T3).
 * timeoutOccurred      — true when the timer won the race (rule T3).
 */
public final class DenoisedFrame {

    private final String burstId;
    private final float[] image;
    private final String modelVariant;
    private final String precision;
    private final long latencyNs;
    private final boolean discardRaceOccurred;
    private final boolean timeoutOccurred;

    public DenoisedFrame(String burstId, float[] image, String modelVariant,
                         String precision, long latencyNs,
                         boolean discardRaceOccurred, boolean timeoutOccurred) {
        this.burstId = burstId;
        this.image = image;
        this.modelVariant = modelVariant;
        this.precision = precision;
        this.latencyNs = latencyNs;
        this.discardRaceOccurred = discardRaceOccurred;
        this.timeoutOccurred = timeoutOccurred;
    }

    public String getBurstId()       { return burstId; }
    public float[] getImage()        { return image; }
    public String getModelVariant()  { return modelVariant; }
    public String getPrecision()     { return precision; }
    public long getLatencyNs()       { return latencyNs; }
    public boolean isDiscardRaceOccurred() { return discardRaceOccurred; }
    public boolean isTimeoutOccurred()     { return timeoutOccurred; }

    /** Wire-format latency in milliseconds (§5.4). */
    public double getLatencyMs() { return latencyNs / 1_000_000.0; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DenoisedFrame)) return false;
        DenoisedFrame that = (DenoisedFrame) o;
        return latencyNs == that.latencyNs
            && discardRaceOccurred == that.discardRaceOccurred
            && timeoutOccurred == that.timeoutOccurred
            && burstId.equals(that.burstId)
            && Arrays.equals(image, that.image)
            && modelVariant.equals(that.modelVariant)
            && precision.equals(that.precision);
    }

    @Override
    public int hashCode() {
        int result = burstId.hashCode();
        result = 31 * result + Arrays.hashCode(image);
        result = 31 * result + modelVariant.hashCode();
        result = 31 * result + precision.hashCode();
        result = 31 * result + Long.hashCode(latencyNs);
        result = 31 * result + Boolean.hashCode(discardRaceOccurred);
        result = 31 * result + Boolean.hashCode(timeoutOccurred);
        return result;
    }
}
