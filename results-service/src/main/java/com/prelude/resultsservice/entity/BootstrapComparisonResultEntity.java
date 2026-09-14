package com.prelude.resultsservice.entity;

import java.time.Instant;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "bootstrap_comparison_result")
public class BootstrapComparisonResultEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    @Column(name = "run_id", nullable = false)
    private String runId;

    @Column(name = "submission_id", nullable = false)
    private UUID submissionId;

    @Column(name = "result_version", nullable = false)
    private int resultVersion;

    @Column(name = "comparison_key", nullable = false)
    private String comparisonKey;

    @Column(name = "metric", nullable = false)
    private String metric;

    @Column(name = "n_images", nullable = false)
    private int nImages;

    @Column(name = "n_comparisons", nullable = false)
    private int nComparisons;

    @Column(name = "corrected_alpha", nullable = false)
    private double correctedAlpha;

    @Column(name = "mean_delta", nullable = false)
    private double meanDelta;

    @Column(name = "ci_lower", nullable = false)
    private double ciLower;

    @Column(name = "ci_upper", nullable = false)
    private double ciUpper;

    @Column(name = "ci_width", nullable = false)
    private double ciWidth;

    @Column(name = "significance_status", nullable = false)
    private String significanceStatus;

    @Column(name = "resamples", nullable = false)
    private int resamples;

    @Column(name = "computed_at", nullable = false)
    private Instant computedAt;

    protected BootstrapComparisonResultEntity() {
    }

    public BootstrapComparisonResultEntity(String runId, UUID submissionId, int resultVersion, String comparisonKey,
                                           String metric, int nImages, int nComparisons, double correctedAlpha,
                                           double meanDelta, double ciLower, double ciUpper, double ciWidth,
                                           String significanceStatus, int resamples, Instant computedAt) {
        this.runId = runId;
        this.submissionId = submissionId;
        this.resultVersion = resultVersion;
        this.comparisonKey = comparisonKey;
        this.metric = metric;
        this.nImages = nImages;
        this.nComparisons = nComparisons;
        this.correctedAlpha = correctedAlpha;
        this.meanDelta = meanDelta;
        this.ciLower = ciLower;
        this.ciUpper = ciUpper;
        this.ciWidth = ciWidth;
        this.significanceStatus = significanceStatus;
        this.resamples = resamples;
        this.computedAt = computedAt;
    }
}