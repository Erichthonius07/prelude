package com.prelude.resultsservice.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;

@Entity
@Table(name = "fusion_strategy_score")
public class FusionStrategyScoreEntity extends AbstractMetricBase {

    @Column(name = "strategy", nullable = false)
    private String strategy;

    @Column(name = "ssim", nullable = false)
    private double ssim;

    @Column(name = "psnr")
    private Double psnr;

    protected FusionStrategyScoreEntity() {
    }

    public FusionStrategyScoreEntity(String runId, UUID submissionId, String imageId, int resultVersion,
                                     String strategy, double ssim, Double psnr) {
        super(runId, submissionId, imageId, resultVersion);
        this.strategy = strategy;
        this.ssim = ssim;
        this.psnr = psnr;
    }
}