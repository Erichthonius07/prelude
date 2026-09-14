package com.prelude.resultsservice.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;

@Entity
@Table(name = "restormer_go_nogo")
public class RestormerGoNogoEntity extends AbstractMetricBase {

    @Column(name = "baseline_int8_psnr", nullable = false)
    private double baselineInt8Psnr;

    @Column(name = "baseline_int8_ssim", nullable = false)
    private double baselineInt8Ssim;

    @Column(name = "baseline_int8_latency_ms")
    private Double baselineInt8LatencyMs;

    @Column(name = "restormer_psnr", nullable = false)
    private double restormerPsnr;

    @Column(name = "restormer_ssim", nullable = false)
    private double restormerSsim;

    @Column(name = "restormer_latency_ms")
    private Double restormerLatencyMs;

    protected RestormerGoNogoEntity() {
    }

    public RestormerGoNogoEntity(String runId, UUID submissionId, String imageId, int resultVersion,
                                 double baselineInt8Psnr, double baselineInt8Ssim, Double baselineInt8LatencyMs,
                                 double restormerPsnr, double restormerSsim, Double restormerLatencyMs) {
        super(runId, submissionId, imageId, resultVersion);
        this.baselineInt8Psnr = baselineInt8Psnr;
        this.baselineInt8Ssim = baselineInt8Ssim;
        this.baselineInt8LatencyMs = baselineInt8LatencyMs;
        this.restormerPsnr = restormerPsnr;
        this.restormerSsim = restormerSsim;
        this.restormerLatencyMs = restormerLatencyMs;
    }
}