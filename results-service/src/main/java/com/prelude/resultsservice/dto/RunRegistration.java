package com.prelude.resultsservice.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;

public record RunRegistration(
        @NotBlank String runId,
        String description,
        String phase,
        @NotNull @Positive Integer plannedTestN,
        @NotNull @Positive Integer plannedComparisons,
        Integer bootstrapResamples,
        Double ciLevel,
        String createdBy) {

    public record Response(RunRegistration run, boolean duplicate) {}
}