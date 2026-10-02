package com.prelude.resultsservice.read.dto;

import java.time.Instant;
import java.util.List;

/** Response/request DTOs for the read & query API (addendum §13). */
public final class ReadDtos {

    private ReadDtos() {
    }

    public record Paged<T>(List<T> content, int page, int size, long totalElements, int totalPages) {
    }

    public record RunSummary(String runId, String description, String phase, Instant createdAt) {
    }

    public record RunDetail(String runId, String description, String phase,
                            int plannedTestN, int plannedComparisons, int bootstrapResamples,
                            double ciLevel, String createdBy, Instant createdAt) {
    }

    /** {@code variant} is set for batch_variant, {@code strategy} for fusion; {@code latencyMs} only for batch_variant. */
    public record ImageScoreRow(String imageId, String variant, String strategy,
                                Double ssim, Double psnr, Double latencyMs, int resultVersion) {
    }

    public record AblationVariant(String variant, long n, Double meanSsim, Double meanPsnr, Double p95LatencyMs) {
    }

    public record AblationResponse(String runId, List<AblationVariant> variants) {
    }

    /** {@code metrics} defaults to psnr+ssim when null/empty; {@code seed} optional; {@code heldOutOnly} defaults false. */
    public record BootstrapComputeRequest(String variantA, String variantB, List<String> metrics,
                                          Long seed, Boolean heldOutOnly) {
    }

    public record BootstrapComputeResponse(String runId, String comparisonKey, List<BootstrapMetricResult> results) {
    }

    public record BootstrapMetricResult(String metric, int pairedN, double meanDelta,
                                        double ciLower, double ciUpper, double ciWidth,
                                        String significanceStatus, int resamples,
                                        int plannedComparisons, double correctedAlpha, long seed,
                                        int resultVersion) {
    }
}