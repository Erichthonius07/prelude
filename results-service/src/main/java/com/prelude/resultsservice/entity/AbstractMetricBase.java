package com.prelude.resultsservice.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.MappedSuperclass;

@MappedSuperclass
public abstract class AbstractMetricBase {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    @Column(name = "run_id", nullable = false)
    private String runId;

    @Column(name = "submission_id", nullable = false)
    private UUID submissionId;

    @Column(name = "image_id", nullable = false)
    private String imageId;

    @Column(name = "result_version", nullable = false)
    private int resultVersion;

    protected AbstractMetricBase() {
    }

    protected AbstractMetricBase(String runId, UUID submissionId, String imageId, int resultVersion) {
        this.runId = runId;
        this.submissionId = submissionId;
        this.imageId = imageId;
        this.resultVersion = resultVersion;
    }

    public String getRunId() {
        return runId;
    }

    public String getImageId() {
        return imageId;
    }

    public int getResultVersion() {
        return resultVersion;
    }
}