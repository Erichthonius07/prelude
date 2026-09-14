package com.prelude.resultsservice.dto;

import java.time.Instant;
import java.util.List;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Positive;

public record BootstrapPayload(@NotEmpty List<@Valid Comparison> comparisons) {

    public record Comparison(
            @NotBlank String comparisonKey,
            @NotBlank String metric,
            @NotNull @Positive Integer nImages,
            @NotNull @Positive Integer nComparisons,
            @NotNull Double meanDelta,
            @NotNull Double ciLower,
            @NotNull Double ciUpper,
            @NotNull Double ciWidth,
            @NotBlank @Pattern(regexp = "significant|directional") String significanceStatus,
            Integer resamples,
            @NotNull Instant computedAt) {}
}
