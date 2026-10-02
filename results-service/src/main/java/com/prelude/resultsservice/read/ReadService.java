package com.prelude.resultsservice.read;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import javax.sql.DataSource;

import org.springframework.dao.DuplicateKeyException;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Service;

import com.prelude.resultsservice.read.ReadApiException.BootstrapNotComputable;
import com.prelude.resultsservice.read.ReadApiException.InvalidReadRequest;
import com.prelude.resultsservice.read.ReadApiException.RunNotFound;
import com.prelude.resultsservice.read.dto.ReadDtos.AblationResponse;
import com.prelude.resultsservice.read.dto.ReadDtos.AblationVariant;
import com.prelude.resultsservice.read.dto.ReadDtos.BootstrapComputeRequest;
import com.prelude.resultsservice.read.dto.ReadDtos.BootstrapComputeResponse;
import com.prelude.resultsservice.read.dto.ReadDtos.BootstrapMetricResult;
import com.prelude.resultsservice.read.dto.ReadDtos.ImageScoreRow;
import com.prelude.resultsservice.read.dto.ReadDtos.Paged;
import com.prelude.resultsservice.read.dto.ReadDtos.RunDetail;
import com.prelude.resultsservice.read.dto.ReadDtos.RunSummary;
import com.prelude.resultsservice.service.Canonicalizer;
import com.prelude.resultsservice.stats.PairedBootstrap;
import com.prelude.resultsservice.stats.PairedBootstrapResult;

/**
 * Read & query API (addendum §13). Uses NamedParameterJdbcTemplate with SQL in this new class so
 * no existing JPA entity or repository is edited.
 *
 * NOTE ON AUTOCONFIGURATION: this service constructs its own NamedParameterJdbcTemplate from the
 * injected DataSource, so it does NOT rely on JdbcTemplate autoconfiguration being active (Boot 4
 * splits autoconfiguration into per-technology modules). Only DataSource availability is required,
 * which is already present since Flyway + JPA boot. The applying agent should still confirm the
 * app starts and these queries execute.
 */
@Service
public class ReadService {

    private static final List<String> ABLATION_VARIANTS = List.of(
            "single_raw_frame", "fusion_naive", "fusion_trimmed", "fusion_confidence_weighted",
            "fusion_learned", "full_pipeline", "single_frame_fallback");
    private static final List<String> METRICS = List.of("psnr", "ssim");
    private static final int MAX_STORE_ATTEMPTS = 3;

    private final NamedParameterJdbcTemplate jdbc;

    public ReadService(DataSource dataSource) {
        this.jdbc = new NamedParameterJdbcTemplate(dataSource);
    }

    // ------------------------------------------------------------------ runs

    public Paged<RunSummary> listRuns(int page, int size) {
        Map<String, Object> params = Map.of("limit", size, "offset", page * size);
        List<RunSummary> content = jdbc.query(
                "SELECT run_id, description, phase, created_at FROM evaluation_run "
                        + "ORDER BY created_at DESC, run_id LIMIT :limit OFFSET :offset",
                params,
                (rs, i) -> new RunSummary(
                        rs.getString("run_id"),
                        rs.getString("description"),
                        rs.getString("phase"),
                        rs.getTimestamp("created_at").toInstant()));
        long total = Optional.ofNullable(
                jdbc.queryForObject("SELECT COUNT(*) FROM evaluation_run", Map.of(), Long.class)).orElse(0L);
        return new Paged<>(content, page, size, total, totalPages(total, size));
    }

    public RunDetail getRun(String runId) {
        List<RunDetail> rows = jdbc.query(
                "SELECT run_id, description, phase, planned_test_n, planned_comparisons, "
                        + "bootstrap_resamples, ci_level, created_by, created_at "
                        + "FROM evaluation_run WHERE run_id = :runId",
                Map.of("runId", runId),
                (rs, i) -> new RunDetail(
                        rs.getString("run_id"), rs.getString("description"), rs.getString("phase"),
                        rs.getInt("planned_test_n"), rs.getInt("planned_comparisons"),
                        rs.getInt("bootstrap_resamples"), rs.getDouble("ci_level"),
                        rs.getString("created_by"), rs.getTimestamp("created_at").toInstant()));
        if (rows.isEmpty()) {
            throw new RunNotFound(runId);
        }
        return rows.get(0);
    }

    // -------------------------------------------------------- per-image scores

    public Paged<ImageScoreRow> imageScores(String runId, String stage, String variant, String strategy,
                                            boolean history, int page, int size) {
        getRun(runId); // 404 if missing
        boolean isBatch = "batch_variant".equals(stage);
        boolean isFusion = "fusion".equals(stage);
        if (!isBatch && !isFusion) {
            throw new InvalidReadRequest("UNKNOWN_STAGE",
                    "stage must be 'batch_variant' or 'fusion' for per-image score reads, got '" + stage + "'");
        }
        String table = isBatch ? "batch_variant_score" : "fusion_strategy_score";
        String discCol = isBatch ? "variant" : "strategy";
        String discValue = isBatch ? variant : strategy;

        StringBuilder where = new StringBuilder("run_id = :runId");
        Map<String, Object> params = new HashMap<>();
        params.put("runId", runId);
        if (discValue != null && !discValue.isBlank()) {
            where.append(" AND ").append(discCol).append(" = :discriminator");
            params.put("discriminator", discValue);
        }
        params.put("limit", size);
        params.put("offset", page * size);

        String latencyExpr = isBatch ? "latency_ms" : "NULL";
        String dataSql;
        String countSql;
        if (history) {
            dataSql = "SELECT image_id, " + discCol + " AS discriminator, ssim, psnr, " + latencyExpr
                    + " AS latency_ms, result_version FROM " + table + " WHERE " + where
                    + " ORDER BY image_id, " + discCol + ", result_version DESC LIMIT :limit OFFSET :offset";
            countSql = "SELECT COUNT(*) FROM " + table + " WHERE " + where;
        } else {
            dataSql = "SELECT DISTINCT ON (image_id, " + discCol + ") image_id, " + discCol
                    + " AS discriminator, ssim, psnr, " + latencyExpr + " AS latency_ms, result_version FROM "
                    + table + " WHERE " + where + " ORDER BY image_id, " + discCol
                    + ", result_version DESC LIMIT :limit OFFSET :offset";
            countSql = "SELECT COUNT(*) FROM (SELECT DISTINCT image_id, " + discCol + " FROM " + table
                    + " WHERE " + where + ") AS pairs";
        }

        final boolean batch = isBatch;
        List<ImageScoreRow> content = jdbc.query(dataSql, params, (rs, i) -> new ImageScoreRow(
                rs.getString("image_id"),
                batch ? rs.getString("discriminator") : null,
                batch ? null : rs.getString("discriminator"),
                rs.getObject("ssim") == null ? null : rs.getDouble("ssim"),
                rs.getObject("psnr") == null ? null : rs.getDouble("psnr"),
                rs.getObject("latency_ms") == null ? null : rs.getDouble("latency_ms"),
                rs.getInt("result_version")));
        long total = Optional.ofNullable(jdbc.queryForObject(countSql, params, Long.class)).orElse(0L);
        return new Paged<>(content, page, size, total, totalPages(total, size));
    }

    // ---------------------------------------------------------------- ablation

    public AblationResponse ablation(String runId) {
        getRun(runId);
        return new AblationResponse(runId, computeAblationVariants(runId));
    }

    private List<AblationVariant> computeAblationVariants(String runId) {
        String sql = "SELECT variant, ssim, psnr, latency_ms FROM ("
                + "SELECT DISTINCT ON (image_id, variant) variant, ssim, psnr, latency_ms "
                + "FROM batch_variant_score WHERE run_id = :runId "
                + "ORDER BY image_id, variant, result_version DESC) AS latest";

        Map<String, Integer> count = new HashMap<>();
        Map<String, Double> ssimSum = new HashMap<>();
        Map<String, Integer> ssimN = new HashMap<>();
        Map<String, Double> psnrSum = new HashMap<>();
        Map<String, Integer> psnrN = new HashMap<>();
        Map<String, List<Double>> latencies = new HashMap<>();

        jdbc.query(sql, Map.of("runId", runId), (rs) -> {
            String v = rs.getString("variant");
            count.merge(v, 1, Integer::sum);
            double ssim = rs.getDouble("ssim");
            if (!rs.wasNull()) {
                ssimSum.merge(v, ssim, Double::sum);
                ssimN.merge(v, 1, Integer::sum);
            }
            double psnr = rs.getDouble("psnr");
            if (!rs.wasNull()) {
                psnrSum.merge(v, psnr, Double::sum);
                psnrN.merge(v, 1, Integer::sum);
            }
            double lat = rs.getDouble("latency_ms");
            if (!rs.wasNull()) {
                latencies.computeIfAbsent(v, k -> new ArrayList<>()).add(lat);
            }
        });

        List<AblationVariant> out = new ArrayList<>();
        for (String v : ABLATION_VARIANTS) {
            int n = count.getOrDefault(v, 0);
            Double meanSsim = ssimN.getOrDefault(v, 0) > 0 ? ssimSum.get(v) / ssimN.get(v) : null;
            Double meanPsnr = psnrN.getOrDefault(v, 0) > 0 ? psnrSum.get(v) / psnrN.get(v) : null;
            Double p95 = p95NearestRank(latencies.getOrDefault(v, List.of()));
            out.add(new AblationVariant(v, n, meanSsim, meanPsnr, p95));
        }
        return out;
    }

    /** Nearest-rank 95th percentile (docs/data-contract.md §5.4). */
    private static Double p95NearestRank(List<Double> values) {
        if (values.isEmpty()) {
            return null;
        }
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        int n = sorted.size();
        int rank = (int) Math.ceil(0.95 * n);
        rank = Math.max(1, Math.min(n, rank));
        return sorted.get(rank - 1);
    }

    // ---------------------------------------------------------------- snapshot

    public Map<String, Object> snapshot(String runId) {
        RunDetail run = getRun(runId);
        Map<String, Object> body = new LinkedHashMap<>();

        Map<String, Object> runMap = new LinkedHashMap<>();
        runMap.put("runId", run.runId());
        runMap.put("description", run.description());
        runMap.put("phase", run.phase());
        runMap.put("plannedTestN", run.plannedTestN());
        runMap.put("plannedComparisons", run.plannedComparisons());
        runMap.put("ciLevel", run.ciLevel());
        runMap.put("createdAt", run.createdAt().toString());
        body.put("run", runMap);

        List<Map<String, Object>> variantMaps = new ArrayList<>();
        for (AblationVariant v : computeAblationVariants(runId)) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("variant", v.variant());
            m.put("n", v.n());
            m.put("meanSsim", v.meanSsim());
            m.put("meanPsnr", v.meanPsnr());
            m.put("p95LatencyMs", v.p95LatencyMs());
            variantMaps.add(m);
        }
        body.put("ablation", variantMaps);

        body.put("bootstrap", bootstrapComparisonMaps(runId));
        body.put("learnedFusionDelta", learnedFusionDeltaMap(runId));
        body.put("restormer", restormerSummaryMap(runId));

        String version = Canonicalizer.sha256Hex(Canonicalizer.write(body));
        body.put("snapshotVersion", version);
        return body;
    }

    private List<Map<String, Object>> bootstrapComparisonMaps(String runId) {
        String sql = "SELECT comparison_key, metric, variant_a, variant_b, paired_n, mean_delta, ci_lower, "
                + "ci_upper, ci_width, significance_status, resamples, planned_comparisons, corrected_alpha, "
                + "seed, computed_at, result_version FROM ("
                + "SELECT DISTINCT ON (comparison_key, metric) * FROM computed_bootstrap_result "
                + "WHERE run_id = :runId ORDER BY comparison_key, metric, result_version DESC) AS latest "
                + "ORDER BY comparison_key, metric";
        List<Map<String, Object>> out = new ArrayList<>();
        jdbc.query(sql, Map.of("runId", runId), (rs) -> {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("comparisonKey", rs.getString("comparison_key"));
            m.put("metric", rs.getString("metric"));
            m.put("variantA", rs.getString("variant_a"));
            m.put("variantB", rs.getString("variant_b"));
            m.put("pairedN", rs.getInt("paired_n"));
            m.put("meanDelta", rs.getDouble("mean_delta"));
            m.put("ciLower", rs.getDouble("ci_lower"));
            m.put("ciUpper", rs.getDouble("ci_upper"));
            m.put("ciWidth", rs.getDouble("ci_width"));
            m.put("significanceStatus", rs.getString("significance_status"));
            m.put("resamples", rs.getInt("resamples"));
            m.put("plannedComparisons", rs.getInt("planned_comparisons"));
            m.put("correctedAlpha", rs.getDouble("corrected_alpha"));
            m.put("seed", rs.getLong("seed"));
            m.put("computedAt", rs.getTimestamp("computed_at").toInstant().toString());
            m.put("resultVersion", rs.getInt("result_version"));
            out.add(m);
        });
        return out;
    }

    private Map<String, Object> learnedFusionDeltaMap(String runId) {
        String sql = "SELECT model_version, train_ssim, val_ssim, train_val_ssim_delta "
                + "FROM fusion_model_diagnostic WHERE run_id = :runId "
                + "ORDER BY result_version DESC, id DESC LIMIT 1";
        List<Map<String, Object>> rows = jdbc.query(sql, Map.of("runId", runId), (rs, i) -> {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("modelVersion", rs.getString("model_version"));
            m.put("trainSsim", rs.getDouble("train_ssim"));
            m.put("valSsim", rs.getDouble("val_ssim"));
            m.put("trainValSsimDelta", rs.getDouble("train_val_ssim_delta"));
            return m;
        });
        return rows.isEmpty() ? null : rows.get(0);
    }

    private Map<String, Object> restormerSummaryMap(String runId) {
        String sql = "SELECT baseline_int8_psnr, baseline_int8_ssim, restormer_psnr, restormer_ssim FROM ("
                + "SELECT DISTINCT ON (image_id) baseline_int8_psnr, baseline_int8_ssim, restormer_psnr, "
                + "restormer_ssim FROM restormer_go_nogo WHERE run_id = :runId "
                + "ORDER BY image_id, result_version DESC) AS latest";
        double[] sums = new double[4];
        int[] ns = new int[4];
        int[] total = {0};
        jdbc.query(sql, Map.of("runId", runId), (rs) -> {
            total[0]++;
            double bPsnr = rs.getDouble("baseline_int8_psnr");
            if (!rs.wasNull()) { sums[0] += bPsnr; ns[0]++; }
            double bSsim = rs.getDouble("baseline_int8_ssim");
            if (!rs.wasNull()) { sums[1] += bSsim; ns[1]++; }
            double rPsnr = rs.getDouble("restormer_psnr");
            if (!rs.wasNull()) { sums[2] += rPsnr; ns[2]++; }
            double rSsim = rs.getDouble("restormer_ssim");
            if (!rs.wasNull()) { sums[3] += rSsim; ns[3]++; }
        });
        if (total[0] == 0) {
            return null;
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("nImages", total[0]);
        m.put("meanBaselineInt8Psnr", ns[0] > 0 ? sums[0] / ns[0] : null);
        m.put("meanBaselineInt8Ssim", ns[1] > 0 ? sums[1] / ns[1] : null);
        m.put("meanRestormerPsnr", ns[2] > 0 ? sums[2] / ns[2] : null);
        m.put("meanRestormerSsim", ns[3] > 0 ? sums[3] / ns[3] : null);
        return m;
    }

    // ------------------------------------------------------- bootstrap compute

    public BootstrapComputeResponse computeBootstrap(String runId, BootstrapComputeRequest request) {
        RunDetail run = getRun(runId); // 404 if missing
        if (request == null) {
            throw new InvalidReadRequest("MALFORMED_REQUEST", "Request body is required.");
        }
        String variantA = request.variantA();
        String variantB = request.variantB();
        if (variantA == null || variantA.isBlank() || variantB == null || variantB.isBlank()) {
            throw new InvalidReadRequest("MISSING_VARIANTS", "variantA and variantB are required.");
        }
        if (!ABLATION_VARIANTS.contains(variantA)) {
            throw new InvalidReadRequest("UNKNOWN_VARIANT", "Unknown variantA '" + variantA + "'.");
        }
        if (!ABLATION_VARIANTS.contains(variantB)) {
            throw new InvalidReadRequest("UNKNOWN_VARIANT", "Unknown variantB '" + variantB + "'.");
        }
        if (variantA.equals(variantB)) {
            throw new InvalidReadRequest("SAME_VARIANT", "variantA and variantB must differ.");
        }

        List<String> metrics = (request.metrics() == null || request.metrics().isEmpty())
                ? METRICS : request.metrics();
        for (String m : metrics) {
            if (!METRICS.contains(m)) {
                throw new InvalidReadRequest("UNKNOWN_METRIC",
                        "metric must be one of " + METRICS + ", got '" + m + "'.");
            }
        }

        boolean heldOutOnly = Boolean.TRUE.equals(request.heldOutOnly());
        String comparisonKey = variantA + "__vs__" + variantB;
        int plannedComparisons = run.plannedComparisons();

        List<BootstrapMetricResult> results = new ArrayList<>();
        for (String metric : metrics) {
            Map<String, Double> scoresA = fetchLatestScores(runId, variantA, metric, heldOutOnly);
            Map<String, Double> scoresB = fetchLatestScores(runId, variantB, metric, heldOutOnly);
            double[] diffs = PairedBootstrap.pairedDifferences(scoresA, scoresB);
            if (diffs.length == 0) {
                throw new BootstrapNotComputable("NO_OVERLAPPING_IMAGES",
                        "No images have a latest-wins " + metric + " score in both '" + variantA
                                + "' and '" + variantB + "'"
                                + (heldOutOnly ? " within the held-out manifest" : "") + ".");
            }
            if (diffs.length < PairedBootstrap.MIN_PAIRED_N) {
                throw new BootstrapNotComputable("INSUFFICIENT_PAIRED_N",
                        "Paired N = " + diffs.length + " is below the required minimum "
                                + PairedBootstrap.MIN_PAIRED_N + ".");
            }
            if (plannedComparisons <= 0) {
                throw new BootstrapNotComputable("INVALID_PLANNED_COMPARISONS",
                        "Run '" + runId + "' has plannedComparisons = " + plannedComparisons
                                + "; it must be positive.");
            }

            long seed = request.seed() != null ? request.seed() : deriveSeed(runId, comparisonKey, metric);
            PairedBootstrapResult result = PairedBootstrap.bootstrap(diffs, plannedComparisons, seed);
            int storedVersion = storeComputedResult(runId, comparisonKey, metric, variantA, variantB, result);
            results.add(new BootstrapMetricResult(metric, result.pairedN(), result.observedMeanDifference(),
                    result.ciLower(), result.ciUpper(), result.ciWidth(), result.significanceStatus(),
                    result.resamples(), result.plannedComparisons(), result.correctedAlpha(), result.seed(),
                    storedVersion));
        }
        return new BootstrapComputeResponse(runId, comparisonKey, results);
    }

    /** Latest-wins per-image score for one variant+metric. {@code metric} is whitelist-validated. */
    private Map<String, Double> fetchLatestScores(String runId, String variant, String metric, boolean heldOutOnly) {
        if (!METRICS.contains(metric)) {
            throw new InvalidReadRequest("UNKNOWN_METRIC", "metric must be one of " + METRICS + ".");
        }
        StringBuilder sql = new StringBuilder()
                .append("SELECT image_id, ").append(metric).append(" AS score FROM (")
                .append("SELECT DISTINCT ON (image_id) image_id, ").append(metric)
                .append(" FROM batch_variant_score WHERE run_id = :runId AND variant = :variant")
                .append(" ORDER BY image_id, result_version DESC) AS latest")
                .append(" WHERE ").append(metric).append(" IS NOT NULL");
        if (heldOutOnly) {
            sql.append(" AND image_id IN (SELECT image_id FROM heldout_test_image)");
        }
        Map<String, Object> params = new HashMap<>();
        params.put("runId", runId);
        params.put("variant", variant);
        Map<String, Double> out = new HashMap<>();
        // Block lambda (void-compatible only) so this binds unambiguously to RowCallbackHandler,
        // NOT ResultSetExtractor<Double> — out.put(...) returns Double, which made the
        // single-expression form ambiguous.
        jdbc.query(sql.toString(), params, rs -> {
            out.put(rs.getString("image_id"), rs.getDouble("score"));
        });
        return out;
    }

    /** Server-assigned result_version with a small retry on concurrent-insert conflict. */
    private int storeComputedResult(String runId, String comparisonKey, String metric,
                                    String variantA, String variantB, PairedBootstrapResult r) {
        for (int attempt = 1; attempt <= MAX_STORE_ATTEMPTS; attempt++) {
            Integer next = jdbc.queryForObject(
                    "SELECT COALESCE(MAX(result_version), 0) + 1 FROM computed_bootstrap_result "
                            + "WHERE run_id = :runId AND comparison_key = :key AND metric = :metric",
                    Map.of("runId", runId, "key", comparisonKey, "metric", metric), Integer.class);
            int version = next == null ? 1 : next;
            try {
                Map<String, Object> p = new HashMap<>();
                p.put("runId", runId);
                p.put("key", comparisonKey);
                p.put("metric", metric);
                p.put("variantA", variantA);
                p.put("variantB", variantB);
                p.put("pairedN", r.pairedN());
                p.put("meanDelta", r.observedMeanDifference());
                p.put("ciLower", r.ciLower());
                p.put("ciUpper", r.ciUpper());
                p.put("ciWidth", r.ciWidth());
                p.put("status", r.significanceStatus());
                p.put("resamples", r.resamples());
                p.put("planned", r.plannedComparisons());
                p.put("alpha", r.correctedAlpha());
                p.put("seed", r.seed());
                p.put("version", version);
                jdbc.update("INSERT INTO computed_bootstrap_result "
                        + "(run_id, comparison_key, metric, variant_a, variant_b, paired_n, mean_delta, "
                        + "ci_lower, ci_upper, ci_width, significance_status, resamples, planned_comparisons, "
                        + "corrected_alpha, seed, result_version) "
                        + "VALUES (:runId, :key, :metric, :variantA, :variantB, :pairedN, :meanDelta, :ciLower, "
                        + ":ciUpper, :ciWidth, :status, :resamples, :planned, :alpha, :seed, :version)", p);
                return version;
            } catch (DuplicateKeyException e) {
                if (attempt == MAX_STORE_ATTEMPTS) {
                    throw new BootstrapNotComputable("CONCURRENT_STORE_CONFLICT",
                            "Could not store the bootstrap result after " + MAX_STORE_ATTEMPTS
                                    + " attempts; retry the request.");
                }
            }
        }
        throw new BootstrapNotComputable("CONCURRENT_STORE_CONFLICT",
                "Could not store the bootstrap result; retry the request.");
    }

    /** Deterministic seed so a recompute of the same comparison is reproducible when no seed is given. */
    private static long deriveSeed(String runId, String comparisonKey, String metric) {
        String s = runId + "\u0000" + comparisonKey + "\u0000" + metric;
        long h = 0xcbf29ce484222325L; // FNV-1a 64-bit
        for (int i = 0; i < s.length(); i++) {
            h ^= s.charAt(i);
            h *= 0x100000001b3L;
        }
        return h;
    }

    private static int totalPages(long total, int size) {
        if (size <= 0) {
            return 0;
        }
        return (int) Math.ceil((double) total / size);
    }
}