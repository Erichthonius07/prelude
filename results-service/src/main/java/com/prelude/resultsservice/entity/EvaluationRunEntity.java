package com.prelude.resultsservice.entity;

import java.time.Instant;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "evaluation_run")
public class EvaluationRunEntity {

    @Id
    @Column(name = "run_id")
    private String runId;

    @Column(name = "description")
    private String description;

    @Column(name = "phase")
    private String phase;

    @Column(name = "planned_test_n", nullable = false)
    private int plannedTestN;

    @Column(name = "planned_comparisons", nullable = false)
    private int plannedComparisons;

    @Column(name = "bootstrap_resamples", nullable = false)
    private int bootstrapResamples;

    @Column(name = "ci_level", nullable = false)
    private double ciLevel;

    @Column(name = "created_by")
    private String createdBy;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    protected EvaluationRunEntity() {
    }

    public EvaluationRunEntity(String runId, String description, String phase, int plannedTestN,
                               int plannedComparisons, int bootstrapResamples, double ciLevel,
                               String createdBy, Instant createdAt) {
        this.runId = runId;
        this.description = description;
        this.phase = phase;
        this.plannedTestN = plannedTestN;
        this.plannedComparisons = plannedComparisons;
        this.bootstrapResamples = bootstrapResamples;
        this.ciLevel = ciLevel;
        this.createdBy = createdBy;
        this.createdAt = createdAt;
    }

    public String getRunId() {
        return runId;
    }

    public int getPlannedTestN() {
        return plannedTestN;
    }

    public int getPlannedComparisons() {
        return plannedComparisons;
    }

    public int getBootstrapResamples() {
        return bootstrapResamples;
    }

    public double getCiLevel() {
        return ciLevel;
    }
}