package com.prelude.resultsservice.dto;

import java.util.List;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.PositiveOrZero;

public record TrainingPayload(
        @NotBlank String model,
        @NotEmpty List<@Valid ImageScore> perImageScores,
        List<@Valid LossCurve> lossCurves) {

    public record ImageScore(
            @NotBlank String imageId,
            @NotNull @PositiveOrZero Integer epoch,
            Integer globalStep,
            @NotBlank String metricName,
            @NotNull Double value,
            @NotBlank @Pattern(regexp = "train|validation") String split) {}

    public record LossCurve(
            @NotBlank String curve,
            @NotEmpty List<@Valid Point> points) {}

    public record Point(
            @NotNull @PositiveOrZero Integer step,
            @NotNull Double value) {}
}