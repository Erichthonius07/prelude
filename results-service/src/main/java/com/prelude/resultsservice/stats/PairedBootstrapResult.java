package com.prelude.resultsservice.stats;

/**
 * Result of a paired bootstrap on a mean difference. Pure data — no framework types.
 *
 * <p>The interval is computed at the Bonferroni-corrected confidence level (family alpha 0.05
 * split across {@code plannedComparisons} comparisons), so the decision rule matches
 * docs/data-contract.md §9: {@code significant} if and only if the corrected interval excludes
 * zero, otherwise {@code directional}.
 */
public record PairedBootstrapResult(
        int pairedN,
        double observedMeanDifference,
        double ciLower,
        double ciUpper,
        double ciWidth,
        int resamples,
        int plannedComparisons,
        double correctedAlpha,
        String significanceStatus,
        long seed) {

    public static final String SIGNIFICANT = "significant";
    public static final String DIRECTIONAL = "directional";
}