package com.prelude.resultsservice.entity;

import java.time.Instant;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "submission")
public class SubmissionEntity {

    @Id
    @Column(name = "submission_id")
    private UUID submissionId;

    @Column(name = "run_id", nullable = false)
    private String runId;

    @Column(name = "stage", nullable = false)
    private String stage;

    @Column(name = "device_id")
    private String deviceId;

    @Column(name = "app_version")
    private String appVersion;

    @Column(name = "occurred_at")
    private Instant occurredAt;

    @Column(name = "received_at", nullable = false)
    private Instant receivedAt;

    @Column(name = "payload_hash", nullable = false, length = 64)
    private String payloadHash;

    @Column(name = "payload_json", nullable = false)
    private String payloadJson;

    @Column(name = "image_count", nullable = false)
    private int imageCount;

    protected SubmissionEntity() {
    }

    public SubmissionEntity(UUID submissionId, String runId, String stage, String deviceId, String appVersion,
                            Instant occurredAt, Instant receivedAt, String payloadHash, String payloadJson,
                            int imageCount) {
        this.submissionId = submissionId;
        this.runId = runId;
        this.stage = stage;
        this.deviceId = deviceId;
        this.appVersion = appVersion;
        this.occurredAt = occurredAt;
        this.receivedAt = receivedAt;
        this.payloadHash = payloadHash;
        this.payloadJson = payloadJson;
        this.imageCount = imageCount;
    }

    public UUID getSubmissionId() {
        return submissionId;
    }

    public String getRunId() {
        return runId;
    }

    public String getStage() {
        return stage;
    }

    public int getImageCount() {
        return imageCount;
    }

    public String getPayloadHash() {
        return payloadHash;
    }
}