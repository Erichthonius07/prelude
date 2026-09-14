package com.prelude.resultsservice.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;

@Entity
@Table(name = "alignment_image_metric")
public class AlignmentImageMetricEntity extends AbstractMetricBase {

    @Column(name = "total_frame_count", nullable = false)
    private int totalFrameCount;

    @Column(name = "aligned_frame_count", nullable = false)
    private int alignedFrameCount;

    @Column(name = "alignment_success", nullable = false)
    private boolean alignmentSuccess;

    @Column(name = "alignment_confidence")
    private Double alignmentConfidence;

    @Column(name = "classifier_predicted_label")
    private String classifierPredictedLabel;

    @Column(name = "classifier_ground_truth_label")
    private String classifierGroundTruthLabel;

    @Column(name = "classifier_correct")
    private Boolean classifierCorrect;

    protected AlignmentImageMetricEntity() {
    }

    public AlignmentImageMetricEntity(String runId, UUID submissionId, String imageId, int resultVersion,
                                      int totalFrameCount, int alignedFrameCount, boolean alignmentSuccess,
                                      Double alignmentConfidence, String classifierPredictedLabel,
                                      String classifierGroundTruthLabel, Boolean classifierCorrect) {
        super(runId, submissionId, imageId, resultVersion);
        this.totalFrameCount = totalFrameCount;
        this.alignedFrameCount = alignedFrameCount;
        this.alignmentSuccess = alignmentSuccess;
        this.alignmentConfidence = alignmentConfidence;
        this.classifierPredictedLabel = classifierPredictedLabel;
        this.classifierGroundTruthLabel = classifierGroundTruthLabel;
        this.classifierCorrect = classifierCorrect;
    }
}