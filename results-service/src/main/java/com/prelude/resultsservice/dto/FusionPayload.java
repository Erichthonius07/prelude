package com.prelude.resultsservice.dto;

import java.util.List;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;

public record FusionPayload(
        @NotEmpty List<@Valid Image> images,
        @Valid LearnedModelDiagnostics learnedModel) {

    public record Image(
            @NotBlank String imageId,
            @NotEmpty List<@Valid StrategyScore> scores) {}

    public record StrategyScore(
            @NotBlank @Pattern(regexp = "naive|trimmed|confidence_weighted|learned") String strategy,
            @NotNull Double ssim,
            Double psnr) {}

    public record LearnedModelDiagnostics(
            @NotBlank String modelVersion,
            @NotNull Double trainSsim,
            @NotNull Double valSsim,
            Double trainValSsimDelta) {}
}