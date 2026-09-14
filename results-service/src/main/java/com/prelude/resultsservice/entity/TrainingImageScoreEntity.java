package com.prelude.resultsservice.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;

@Entity
@Table(name = "training_image_score")
public class TrainingImageScoreEntity extends AbstractMetricBase {

    @Column(name = "model", nullable = false)
    private String model;

    @Column(name = "epoch", nullable = false)
    private int epoch;

    @Column(name = "global_step")
    private Integer globalStep;

    @Column(name = "metric_name", nullable = false)
    private String metricName;

    @Column(name = "value", nullable = false)
    private double value;

    @Column(name = "split", nullable = false)
    private String split;

    protected TrainingImageScoreEntity() {
    }

    public TrainingImageScoreEntity(String runId, UUID submissionId, String imageId, int resultVersion,
                                    String model, int epoch, Integer globalStep, String metricName,
                                    double value, String split) {
        super(runId, submissionId, imageId, resultVersion);
        this.model = model;
        this.epoch = epoch;
        this.globalStep = globalStep;
        this.metricName = metricName;
        this.value = value;
        this.split = split;
    }
}