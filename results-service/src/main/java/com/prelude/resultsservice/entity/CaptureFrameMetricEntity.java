package com.prelude.resultsservice.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;

@Entity
@Table(name = "capture_frame_metric")
public class CaptureFrameMetricEntity extends AbstractMetricBase {

    @Column(name = "frame_index", nullable = false)
    private int frameIndex;

    @Column(name = "iso", nullable = false)
    private int iso;

    @Column(name = "exposure_time_ns", nullable = false)
    private long exposureTimeNs;

    @Column(name = "captured_at_epoch_ms", nullable = false)
    private long capturedAtEpochMs;

    @Column(name = "sharpness_score", nullable = false)
    private double sharpnessScore;

    @Column(name = "blur_rejected", nullable = false)
    private boolean blurRejected;

    @Column(name = "emergency_fallback", nullable = false)
    private boolean emergencyFallback;

    protected CaptureFrameMetricEntity() {
    }

    public CaptureFrameMetricEntity(String runId, UUID submissionId, String imageId, int resultVersion,
                                    int frameIndex, int iso, long exposureTimeNs, long capturedAtEpochMs,
                                    double sharpnessScore, boolean blurRejected, boolean emergencyFallback) {
        super(runId, submissionId, imageId, resultVersion);
        this.frameIndex = frameIndex;
        this.iso = iso;
        this.exposureTimeNs = exposureTimeNs;
        this.capturedAtEpochMs = capturedAtEpochMs;
        this.sharpnessScore = sharpnessScore;
        this.blurRejected = blurRejected;
        this.emergencyFallback = emergencyFallback;
    }
}