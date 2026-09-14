package com.prelude.resultsservice.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;

@Entity
@Table(name = "denoise_image_metric")
public class DenoiseImageMetricEntity extends AbstractMetricBase {

    @Column(name = "model_variant", nullable = false)
    private String modelVariant;

    @Column(name = "precision", nullable = false)
    private String precision;

    @Column(name = "psnr", nullable = false)
    private double psnr;

    @Column(name = "ssim", nullable = false)
    private double ssim;

    @Column(name = "latency_ms", nullable = false)
    private double latencyMs;

    @Column(name = "discard_race_event", nullable = false)
    private boolean discardRaceEvent;

    @Column(name = "timeout_event", nullable = false)
    private boolean timeoutEvent;

    @Column(name = "model_file_size_bytes")
    private Long modelFileSizeBytes;

    protected DenoiseImageMetricEntity() {
    }

    public DenoiseImageMetricEntity(String runId, UUID submissionId, String imageId, int resultVersion,
                                    String modelVariant, String precision, double psnr, double ssim,
                                    double latencyMs, boolean discardRaceEvent, boolean timeoutEvent,
                                    Long modelFileSizeBytes) {
        super(runId, submissionId, imageId, resultVersion);
        this.modelVariant = modelVariant;
        this.precision = precision;
        this.psnr = psnr;
        this.ssim = ssim;
        this.latencyMs = latencyMs;
        this.discardRaceEvent = discardRaceEvent;
        this.timeoutEvent = timeoutEvent;
        this.modelFileSizeBytes = modelFileSizeBytes;
    }
}