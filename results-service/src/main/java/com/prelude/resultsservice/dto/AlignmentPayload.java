package com.prelude.resultsservice.dto;

import java.util.List;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.PositiveOrZero;

public record AlignmentPayload(
        @NotEmpty List<@Valid Image> images,
        Double heldOutClassifierAccuracy,
        Double heldOutClassifierPrecision,
        Double heldOutClassifierRecall) {

    public record Image(
            @NotBlank String imageId,
            @NotNull Integer totalFrameCount,
            @NotNull Integer alignedFrameCount,
            @NotNull Boolean alignmentSuccess,
            Double alignmentConfidence,
            String classifierPredictedLabel,
            String classifierGroundTruthLabel,
            Boolean classifierCorrect,
            @NotEmpty List<@Valid Frame> frames) {}

    public record Frame(
            @NotNull @PositiveOrZero Integer frameIndex,
            @NotNull Double ransacInlierRatio,
            @NotNull Boolean frameAligned) {}
}