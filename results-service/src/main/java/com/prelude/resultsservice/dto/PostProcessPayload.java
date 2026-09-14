package com.prelude.resultsservice.dto;

import java.util.List;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;

public record PostProcessPayload(@NotEmpty List<@Valid Image> images) {

    public record Image(
            @NotBlank String imageId,
            Double ssimBeforePostprocess,
            Double ssimAfterPostprocess,
            @NotNull Double ssimContribution,
            Double fullPipelineSsim) {}
}
