package com.prelude.resultsservice.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;

@Entity
@Table(name = "alignment_frame_metric")
public class AlignmentFrameMetricEntity extends AbstractMetricBase {

    @Column(name = "frame_index", nullable = false)
    private int frameIndex;

    @Column(name = "ransac_inlier_ratio", nullable = false)
    private double ransacInlierRatio;

    @Column(name = "frame_aligned", nullable = false)
    private boolean frameAligned;

    protected AlignmentFrameMetricEntity() {
    }

    public AlignmentFrameMetricEntity(String runId, UUID submissionId, String imageId, int resultVersion,
                                      int frameIndex, double ransacInlierRatio, boolean frameAligned) {
        super(runId, submissionId, imageId, resultVersion);
        this.frameIndex = frameIndex;
        this.ransacInlierRatio = ransacInlierRatio;
        this.frameAligned = frameAligned;
    }
}