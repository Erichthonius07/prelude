package com.prelude.resultsservice.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "alignment_classifier_summary")
public class AlignmentClassifierSummaryEntity {

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

    @Column(name = "accuracy")
    private Double accuracy;

    @Column(name = "precision")
    private Double precision;

    @Column(name = "recall")
    private Double recall;

    @Column(name = "n_images", nullable = false)
    private int nImages;

    protected AlignmentClassifierSummaryEntity() {
    }

    public AlignmentClassifierSummaryEntity(String runId, UUID submissionId, int resultVersion,
                                            Double accuracy, Double precision, Double recall, int nImages) {
        this.runId = runId;
        this.submissionId = submissionId;
        this.resultVersion = resultVersion;
        this.accuracy = accuracy;
        this.precision = precision;
        this.recall = recall;
        this.nImages = nImages;
    }
}