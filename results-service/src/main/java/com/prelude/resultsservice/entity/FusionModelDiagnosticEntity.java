package com.prelude.resultsservice.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "fusion_model_diagnostic")
public class FusionModelDiagnosticEntity {

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

    @Column(name = "model_version", nullable = false)
    private String modelVersion;

    @Column(name = "train_ssim", nullable = false)
    private double trainSsim;

    @Column(name = "val_ssim", nullable = false)
    private double valSsim;

    @Column(name = "train_val_ssim_delta", nullable = false)
    private double trainValSsimDelta;

    protected FusionModelDiagnosticEntity() {
    }

    public FusionModelDiagnosticEntity(String runId, UUID submissionId, int resultVersion, String modelVersion,
                                       double trainSsim, double valSsim, double trainValSsimDelta) {
        this.runId = runId;
        this.submissionId = submissionId;
        this.resultVersion = resultVersion;
        this.modelVersion = modelVersion;
        this.trainSsim = trainSsim;
        this.valSsim = valSsim;
        this.trainValSsimDelta = trainValSsimDelta;
    }
}