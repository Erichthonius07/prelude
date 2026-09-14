package com.prelude.resultsservice.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;

@Entity
@Table(name = "batch_variant_score")
public class BatchVariantScoreEntity extends AbstractMetricBase {

    @Column(name = "variant", nullable = false)
    private String variant;

    @Column(name = "ssim", nullable = false)
    private double ssim;

    @Column(name = "psnr")
    private Double psnr;

    @Column(name = "latency_ms")
    private Double latencyMs;

    protected BatchVariantScoreEntity() {
    }

    public BatchVariantScoreEntity(String runId, UUID submissionId, String imageId, int resultVersion,
                                   String variant, double ssim, Double psnr, Double latencyMs) {
        super(runId, submissionId, imageId, resultVersion);
        this.variant = variant;
        this.ssim = ssim;
        this.psnr = psnr;
        this.latencyMs = latencyMs;
    }
}