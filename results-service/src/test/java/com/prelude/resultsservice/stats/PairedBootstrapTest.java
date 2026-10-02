package com.prelude.resultsservice.stats;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.junit.jupiter.api.Test;

/**
 * Property tests for the pure-Java paired bootstrap (addendum v1.1.0 §13.7). No Spring context:
 * the class under test has no framework dependencies, so these run anywhere JUnit runs.
 */
class PairedBootstrapTest {

    private static final double EPS = 1e-9;

    // -- pairedDifferences ---------------------------------------------------

    @Test
    void pairedDifferencesPairsOnlyCommonImagesOrderedByImageId() {
        Map<String, Double> a = mapOf("img2", 0.5, "img1", 0.75, "img3", 0.25);
        Map<String, Double> b = mapOf("img1", 0.25, "img2", 0.75, "img4", 0.5);

        // img3 (only in a) and img4 (only in b) are dropped; order is lexicographic by imageId;
        // each difference is a - b. 0.75-0.25 and 0.5-0.75 are exact in binary floating point.
        assertArrayEquals(new double[] { 0.5, -0.25 }, PairedBootstrap.pairedDifferences(a, b), 0.0);
    }

    @Test
    void pairedDifferencesIgnoresNullScoresOnEitherSide() {
        Map<String, Double> a = mapOf("img1", 0.5, "img2", null, "img3", 0.25);
        Map<String, Double> b = mapOf("img1", null, "img2", 0.5, "img3", 0.75);

        // Only img3 has a non-null score on both sides.
        assertArrayEquals(new double[] { -0.5 }, PairedBootstrap.pairedDifferences(a, b), 0.0);
    }

    @Test
    void pairedDifferencesRejectsNullScoreMaps() {
        assertThrows(IllegalArgumentException.class, () -> PairedBootstrap.pairedDifferences(null, Map.of()));
        assertThrows(IllegalArgumentException.class, () -> PairedBootstrap.pairedDifferences(Map.of(), null));
    }

    // -- bootstrap input validation ------------------------------------------

    @Test
    void bootstrapRejectsNullEmptyAndUnderSizedDifferences() {
        assertThrows(IllegalArgumentException.class, () -> PairedBootstrap.bootstrap(null, 1, 1L));
        assertThrows(IllegalArgumentException.class, () -> PairedBootstrap.bootstrap(new double[0], 1, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> PairedBootstrap.bootstrap(new double[] { 0.1, -0.1, 0.2, -0.2 }, 1, 1L));
    }

    @Test
    void bootstrapRejectsNonPositivePlannedComparisons() {
        double[] diffs = { 0.5, -0.25, 0.125, -0.125, 0.25 };
        assertThrows(IllegalArgumentException.class, () -> PairedBootstrap.bootstrap(diffs, 0, 1L));
        assertThrows(IllegalArgumentException.class, () -> PairedBootstrap.bootstrap(diffs, -3, 1L));
    }

    // -- determinism & the seeded-RNG algorithm -------------------------------

    @Test
    void sameSeedReproducesIdenticalResult() {
        double[] diffs = { 0.5, -0.25, 0.125, -0.375, 0.25, 0.0, 0.125 };

        PairedBootstrapResult first = PairedBootstrap.bootstrap(diffs, 2, 42L);
        PairedBootstrapResult second = PairedBootstrap.bootstrap(diffs, 2, 42L);

        assertEquals(first, second);
        assertEquals(first.observedMeanDifference(), second.observedMeanDifference(), 0.0);
        assertEquals(first.ciLower(), second.ciLower(), 0.0);
        assertEquals(first.ciUpper(), second.ciUpper(), 0.0);
        assertEquals(first.significanceStatus(), second.significanceStatus());
    }

    @Test
    void matchesIndependentReferenceImplementation() {
        double[] diffs = { 0.5, -0.25, 0.125, -0.375, 0.25, 0.0, 0.125, -0.125 };
        long seed = 7L;
        int comparisons = 4;

        PairedBootstrapResult result = PairedBootstrap.bootstrap(diffs, comparisons, seed);

        // Independent re-derivation of the documented algorithm: JDK-specified Random(seed),
        // 10k resample means, numpy-style linear-interpolation quantiles at the corrected level.
        double correctedAlpha = PairedBootstrap.FAMILY_ALPHA / comparisons;
        int n = diffs.length;
        Random rng = new Random(seed);
        double[] means = new double[PairedBootstrap.RESAMPLES];
        for (int r = 0; r < PairedBootstrap.RESAMPLES; r++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += diffs[rng.nextInt(n)];
            }
            means[r] = sum / n;
        }
        Arrays.sort(means);
        double lower = linearQuantile(means, correctedAlpha / 2.0);
        double upper = linearQuantile(means, 1.0 - (correctedAlpha / 2.0));

        double observed = Arrays.stream(diffs).sum() / n;
        assertEquals(n, result.pairedN());
        assertEquals(observed, result.observedMeanDifference(), 0.0);
        assertEquals(lower, result.ciLower(), 0.0);
        assertEquals(upper, result.ciUpper(), 0.0);
        assertEquals(PairedBootstrap.RESAMPLES, result.resamples());
        assertEquals(comparisons, result.plannedComparisons());
        assertEquals(correctedAlpha, result.correctedAlpha(), 0.0);
        assertEquals(seed, result.seed());
    }

    // -- statistical properties ----------------------------------------------

    @Test
    void constantDifferencesYieldDegenerateIntervalAndDecideStatus() {
        double[] positive = { 0.25, 0.25, 0.25, 0.25, 0.25, 0.25 };
        PairedBootstrapResult sig = PairedBootstrap.bootstrap(positive, 1, 3L);
        assertEquals(0.25, sig.observedMeanDifference(), 0.0);
        assertEquals(0.25, sig.ciLower(), 0.0);
        assertEquals(0.25, sig.ciUpper(), 0.0);
        assertEquals(0.0, sig.ciWidth(), 0.0);
        assertEquals(PairedBootstrapResult.SIGNIFICANT, sig.significanceStatus());

        double[] zeros = new double[8];
        PairedBootstrapResult dir = PairedBootstrap.bootstrap(zeros, 1, 3L);
        assertEquals(0.0, dir.ciLower(), 0.0);
        assertEquals(0.0, dir.ciUpper(), 0.0);
        assertEquals(PairedBootstrapResult.DIRECTIONAL, dir.significanceStatus());
    }

    @Test
    void intervalWidensWithBonferroniCorrection() {
        double[] diffs = { 0.5, -0.25, 0.125, -0.375, 0.25, 0.0, 0.125, -0.125, 0.375, -0.0625 };
        long seed = 11L;

        PairedBootstrapResult c1 = PairedBootstrap.bootstrap(diffs, 1, seed);
        PairedBootstrapResult c5 = PairedBootstrap.bootstrap(diffs, 5, seed);
        PairedBootstrapResult c25 = PairedBootstrap.bootstrap(diffs, 25, seed);

        assertEquals(0.05, c1.correctedAlpha(), 0.0);
        assertEquals(0.05 / 5, c5.correctedAlpha(), 0.0);
        assertEquals(0.05 / 25, c25.correctedAlpha(), 0.0);
        // Same seed and N reproduce the same resample means, so a stricter family only widens
        // the quantile window: the interval is monotone non-decreasing in plannedComparisons.
        assertTrue(c1.ciWidth() <= c5.ciWidth());
        assertTrue(c5.ciWidth() <= c25.ciWidth());
        assertTrue(c25.ciWidth() > c1.ciWidth());
    }

    @Test
    void swappingVariantsMirrorsDifferenceAndInterval() {
        Map<String, Double> a = mapOf("img1", 0.5, "img2", 0.75, "img3", 0.25,
                "img4", 0.625, "img5", 0.375, "img6", 0.875);
        Map<String, Double> b = mapOf("img1", 0.375, "img2", 0.5, "img3", 0.4375,
                "img4", 0.25, "img5", 0.3125, "img6", 0.5625);

        PairedBootstrapResult ab = PairedBootstrap.computeFromScores(a, b, 2, 99L);
        PairedBootstrapResult ba = PairedBootstrap.computeFromScores(b, a, 2, 99L);

        assertEquals(ab.observedMeanDifference(), -ba.observedMeanDifference(), 0.0);
        assertEquals(ab.ciLower(), -ba.ciUpper(), EPS);
        assertEquals(ab.ciUpper(), -ba.ciLower(), EPS);
        assertEquals(ab.significanceStatus(), ba.significanceStatus());
        assertEquals(ab.pairedN(), ba.pairedN());
    }

    @Test
    void computeFromScoresAgreesWithBootstrapOfPairedDifferences() {
        Map<String, Double> a = mapOf("img1", 0.5, "img2", 0.75, "img3", 0.25, "img4", 0.625, "img5", 0.375);
        Map<String, Double> b = mapOf("img1", 0.375, "img2", 0.5, "img3", 0.4375, "img4", 0.25, "img5", 0.3125);
        double[] diffs = PairedBootstrap.pairedDifferences(a, b);

        PairedBootstrapResult fromScores = PairedBootstrap.computeFromScores(a, b, 3, 123L);
        PairedBootstrapResult fromDiffs = PairedBootstrap.bootstrap(diffs, 3, 123L);

        assertEquals(fromDiffs, fromScores);
        assertEquals(5, fromScores.pairedN());
        assertEquals(10_000, fromScores.resamples());
        assertEquals(3, fromScores.plannedComparisons());
        assertEquals(123L, fromScores.seed());
    }

    // -- helpers --------------------------------------------------------------

    /** numpy "linear" quantile over an ascending-sorted array (independent copy for cross-checks). */
    private static double linearQuantile(double[] sortedAsc, double p) {
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

    private static Map<String, Double> mapOf(Object... pairs) {
        Map<String, Double> map = new HashMap<>();
        for (int i = 0; i < pairs.length; i += 2) {
            map.put((String) pairs[i], (Double) pairs[i + 1]);
        }
        return map;
    }

    private static void assertArrayEquals(double[] expected, double[] actual, double delta) {
        org.junit.jupiter.api.Assertions.assertArrayEquals(expected, actual, delta);
    }
}
