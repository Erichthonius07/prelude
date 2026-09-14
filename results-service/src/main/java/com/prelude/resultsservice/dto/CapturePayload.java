package com.prelude.resultsservice.dto;

import java.util.List;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.PositiveOrZero;

public record CapturePayload(@NotEmpty List<@Valid Burst> bursts) {

    public record Burst(
            @NotBlank String imageId,
            @NotEmpty List<@Valid Frame> frames) {}

    public record Frame(
            @NotNull @PositiveOrZero Integer frameIndex,
            @NotNull Integer iso,
            @NotNull Long exposureTimeNs,
            @NotNull Long capturedAtEpochMs,
            @NotNull Double sharpnessScore,
            boolean blurRejected,
            boolean emergencyFallback) {}
}