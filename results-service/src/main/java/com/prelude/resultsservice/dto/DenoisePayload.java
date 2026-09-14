package com.prelude.resultsservice.dto;

import java.util.List;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;

public record DenoisePayload(
        @NotBlank String modelVariant,
        Long modelFileSizeBytes,
        @NotEmpty List<@Valid Image> images,
        List<@Valid RestormerComparison> restormerComparisons) {

    public record Image(
            @NotBlank String imageId,
            @NotBlank @Pattern(regexp = "fp32|int8") String precision,
            @NotNull Double psnr,
            @NotNull Double ssim,
            @NotNull Double latencyMs,
            boolean discardRaceEvent,
            boolean timeoutEvent) {}

    public record RestormerComparison(
            @NotBlank String imageId,
            @NotNull Double baselineInt8Psnr,
            @NotNull Double baselineInt8Ssim,
            Double baselineInt8LatencyMs,
            @NotNull Double restormerPsnr,
            @NotNull Double restormerSsim,
            Double restormerLatencyMs) {}
}