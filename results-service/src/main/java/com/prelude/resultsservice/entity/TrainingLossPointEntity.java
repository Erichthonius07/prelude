package com.prelude.resultsservice.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.util.UUID;

@Entity
@Table(name = "training_loss_point")
public class TrainingLossPointEntity {

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

    @Column(name = "model", nullable = false)
    private String model;

    @Column(name = "curve", nullable = false)
    private String curve;

    @Column(name = "step", nullable = false)
    private int step;

    @Column(name = "value", nullable = false)
    private double value;

    protected TrainingLossPointEntity() {
    }

    public TrainingLossPointEntity(String runId, UUID submissionId, int resultVersion,
                                   String model, String curve, int step, double value) {
        this.runId = runId;
        this.submissionId = submissionId;
        this.resultVersion = resultVersion;
        this.model = model;
        this.curve = curve;
        this.step = step;
        this.value = value;
    }
}