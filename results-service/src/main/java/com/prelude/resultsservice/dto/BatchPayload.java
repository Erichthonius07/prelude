package com.prelude.resultsservice.dto;

import java.util.List;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;

public record BatchPayload(@NotEmpty List<@Valid Score> scores) {

    public record Score(
            @NotBlank String imageId,
            @NotBlank String variant,
            @NotNull Double ssim,
            Double psnr,
            Double latencyMs) {}
}