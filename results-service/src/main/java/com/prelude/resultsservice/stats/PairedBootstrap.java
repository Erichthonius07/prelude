package com.prelude.resultsservice.stats;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Paired bootstrap on a mean difference. Pure Java — no Spring, JPA or JDBC imports, so it
 * compiles and tests anywhere.
 *
 * <p>Reproducibility: uses {@link java.util.Random}, whose sequence is fully specified by the
 * JDK, so the same seed reproduces the same result on any JVM. Paired differences are ordered
 * by imageId (lexicographic) before resampling, so the same two score maps and the same seed
 * always produce the same result regardless of map iteration order.
 */
public final class PairedBootstrap {

    public static final int RESAMPLES = 10_000;
    public static final double FAMILY_ALPHA = 0.05;

    /**
     * Minimum number of paired images required to run the bootstrap.
     * DECISION FOR THE ORCHESTRATOR — default chosen conservatively; change here if a different
     * floor is ratified.
     */
    public static final int MIN_PAIRED_N = 5;

    private PairedBootstrap() {
    }

    /**
     * Pairs two per-image score maps by imageId, keeping only images present with a non-null
     * score in BOTH maps. Differences are {@code a - b}, ordered by imageId for determinism.
     */
    public static double[] pairedDifferences(Map<String, Double> a, Map<String, Double> b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("score maps must not be null");
        }
        List<String> common = new ArrayList<>();
        for (Map.Entry<String, Double> e : a.entrySet()) {
            String id = e.getKey();
            if (e.getValue() != null && b.get(id) != null) {
                common.add(id);
            }
        }
        Collections.sort(common);
        double[] d = new double[common.size()];
        for (int i = 0; i < common.size(); i++) {
            d[i] = a.get(common.get(i)) - b.get(common.get(i));
        }
        return d;
    }

    /**
     * Runs the paired bootstrap on precomputed differences and returns the corrected-level
     * interval plus significance status.
     *
     * @throws IllegalArgumentException on zero differences, fewer than {@link #MIN_PAIRED_N}
     *         differences, or a non-positive {@code plannedComparisons}.
     */
    public static PairedBootstrapResult bootstrap(double[] differences, int plannedComparisons, long seed) {
        if (differences == null) {
            throw new IllegalArgumentException("differences must not be null");
        }
        int n = differences.length;
        if (n == 0) {
            throw new IllegalArgumentException("no overlapping images: zero paired differences");
        }
        if (n < MIN_PAIRED_N) {
            throw new IllegalArgumentException(
                    "insufficient paired N: " + n + " < required minimum " + MIN_PAIRED_N);
        }
        if (plannedComparisons <= 0) {
            throw new IllegalArgumentException(
                    "plannedComparisons must be a positive integer, got " + plannedComparisons);
        }

        double correctedAlpha = FAMILY_ALPHA / plannedComparisons;
        double observed = mean(differences);

        Random rng = new Random(seed); // JDK-specified LCG: same seed, same sequence, any JVM
        double[] means = new double[RESAMPLES];
        for (int r = 0; r < RESAMPLES; r++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += differences[rng.nextInt(n)];
            }
            means[r] = sum / n;
        }
        Arrays.sort(means);

        double lower = quantile(means, correctedAlpha / 2.0);
        double upper = quantile(means, 1.0 - (correctedAlpha / 2.0));
        double ciWidth = upper - lower;
        boolean excludesZero = lower > 0.0 || upper < 0.0;
        String status = excludesZero ? PairedBootstrapResult.SIGNIFICANT : PairedBootstrapResult.DIRECTIONAL;

        return new PairedBootstrapResult(n, observed, lower, upper, ciWidth, RESAMPLES,
                plannedComparisons, correctedAlpha, status, seed);
    }

    /** Convenience: pair two score maps by imageId and bootstrap in one call. */
    public static PairedBootstrapResult computeFromScores(Map<String, Double> a, Map<String, Double> b,
                                                          int plannedComparisons, long seed) {
        return bootstrap(pairedDifferences(a, b), plannedComparisons, seed);
    }

    private static double mean(double[] values) {
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.length;
    }

    /** Linear-interpolation quantile (numpy "linear" method) over an ascending-sorted array. */
    static double quantile(double[] sortedAsc, double p) {
        int n = sortedAsc.length;
        if (n == 1) {
            return sortedAsc[0];
        }
        double pos = p * (n - 1);
        int lo = (int) Math.floor(pos);
        int hi = (int) Math.ceil(pos);
        if (lo == hi) {
            return sortedAsc[lo];
        }
        double frac = pos - lo;
        return sortedAsc[lo] + frac * (sortedAsc[hi] - sortedAsc[lo]);
    }
}