package com.prelude.resultsservice.entity;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;

@Entity
@Table(name = "postprocess_image_metric")
public class PostProcessImageMetricEntity extends AbstractMetricBase {

    @Column(name = "ssim_before_postprocess")
    private Double ssimBeforePostprocess;

    @Column(name = "ssim_after_postprocess")
    private Double ssimAfterPostprocess;

    @Column(name = "ssim_contribution", nullable = false)
    private double ssimContribution;

    @Column(name = "full_pipeline_ssim")
    private Double fullPipelineSsim;

    protected PostProcessImageMetricEntity() {
    }

    public PostProcessImageMetricEntity(String runId, UUID submissionId, String imageId, int resultVersion,
                                        Double ssimBeforePostprocess, Double ssimAfterPostprocess,
                                        double ssimContribution, Double fullPipelineSsim) {
        super(runId, submissionId, imageId, resultVersion);
        this.ssimBeforePostprocess = ssimBeforePostprocess;
        this.ssimAfterPostprocess = ssimAfterPostprocess;
        this.ssimContribution = ssimContribution;
        this.fullPipelineSsim = fullPipelineSsim;
    }
}