package com.prelude.denoise.model;

import java.util.Arrays;
import java.util.Map;

/**
 * Input to the denoise module. Matches contract §6.3.
 *
 * burstId      — becomes imageId for all downstream reporting (§2.3).
 * strategy     — which fusion strategy produced this frame.
 * image        — linear-domain fused frame pixel data (like a flat numpy float32 array).
 * pipelineMode — fusion_multi | fusion_single (§6.6).
 *                MUST NOT influence the denoise timeout (§6.6, rules D4/T5).
 * evaluationSidecars — present in evaluation mode only (§6.3).
 */
public final class FusedFrame {

    private final String burstId;
    private final String strategy;
    private final float[] image;
    private final String pipelineMode;
    private final Map<String, float[]> evaluationSidecars;  // nullable

    public FusedFrame(String burstId, String strategy, float[] image,
                      String pipelineMode, Map<String, float[]> evaluationSidecars) {
        this.burstId = burstId;
        this.strategy = strategy;
        this.image = image;
        this.pipelineMode = pipelineMode;
        this.evaluationSidecars = evaluationSidecars;
    }

    public FusedFrame(String burstId, String strategy, float[] image, String pipelineMode) {
        this(burstId, strategy, image, pipelineMode, null);
    }

    public String getBurstId()    { return burstId; }
    public String getStrategy()   { return strategy; }
    public float[] getImage()     { return image; }
    public String getPipelineMode() { return pipelineMode; }
    public Map<String, float[]> getEvaluationSidecars() { return evaluationSidecars; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof FusedFrame)) return false;
        FusedFrame that = (FusedFrame) o;
        return burstId.equals(that.burstId)
            && strategy.equals(that.strategy)
            && Arrays.equals(image, that.image)
            && pipelineMode.equals(that.pipelineMode);
    }

    @Override
    public int hashCode() {
        int result = burstId.hashCode();
        result = 31 * result + strategy.hashCode();
        result = 31 * result + Arrays.hashCode(image);
        result = 31 * result + pipelineMode.hashCode();
        return result;
    }
}
