package com.prelude.resultsservice.service;

import com.prelude.resultsservice.api.PreludeApiException;
import com.prelude.resultsservice.dto.AlignmentPayload;
import com.prelude.resultsservice.dto.BatchPayload;
import com.prelude.resultsservice.dto.BootstrapPayload;
import com.prelude.resultsservice.dto.CapturePayload;
import com.prelude.resultsservice.dto.DenoisePayload;
import com.prelude.resultsservice.dto.FusionPayload;
import com.prelude.resultsservice.dto.IngestionEnvelope;
import com.prelude.resultsservice.dto.IngestionResponse;
import com.prelude.resultsservice.dto.PostProcessPayload;
import com.prelude.resultsservice.dto.RunRegistration;
import com.prelude.resultsservice.dto.TrainingPayload;
import com.prelude.resultsservice.entity.AlignmentClassifierSummaryEntity;
import com.prelude.resultsservice.entity.AlignmentFrameMetricEntity;
import com.prelude.resultsservice.entity.AlignmentImageMetricEntity;
import com.prelude.resultsservice.entity.BatchVariantScoreEntity;
import com.prelude.resultsservice.entity.BootstrapComparisonResultEntity;
import com.prelude.resultsservice.entity.CaptureFrameMetricEntity;
import com.prelude.resultsservice.entity.DenoiseImageMetricEntity;
import com.prelude.resultsservice.entity.EvaluationRunEntity;
import com.prelude.resultsservice.entity.FusionModelDiagnosticEntity;
import com.prelude.resultsservice.entity.FusionStrategyScoreEntity;
import com.prelude.resultsservice.entity.PostProcessImageMetricEntity;
import com.prelude.resultsservice.entity.RestormerGoNogoEntity;
import com.prelude.resultsservice.entity.SubmissionEntity;
import com.prelude.resultsservice.entity.TrainingImageScoreEntity;
import com.prelude.resultsservice.entity.TrainingLossPointEntity;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.PersistenceException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.UUID;

@Service
public class IngestionService {

    public static final int REQUIRED_BOOTSTRAP_RESAMPLES = 10_000;
    public static final double DEFAULT_CI_LEVEL = 0.95;

    private static final List<String> REQUIRED_FUSION_STRATEGIES =
            List.of("naive", "trimmed", "confidence_weighted");

    /** The spec's evaluation table, verbatim (ratified — contract §5.7). */
    private static final Set<String> KNOWN_BATCH_VARIANTS = Set.of(
            "single_raw_frame", "fusion_naive", "fusion_trimmed", "fusion_confidence_weighted",
            "fusion_learned", "full_pipeline", "single_frame_fallback");

    @PersistenceContext
    private EntityManager em;

    // ------------------------------------------------------------------ runs

    @Transactional
    public RunRegistration.Response registerRun(RunRegistration req) {
        int resamples = req.bootstrapResamples() == null
                ? REQUIRED_BOOTSTRAP_RESAMPLES
                : req.bootstrapResamples().intValue();
        double ciLevel = req.ciLevel() == null ? DEFAULT_CI_LEVEL : req.ciLevel().doubleValue();
        if (resamples != REQUIRED_BOOTSTRAP_RESAMPLES) {
            throw new PreludeApiException.ContractViolation("UNSUPPORTED_RESAMPLE_COUNT",
                    "Bootstrap resample count is fixed at " + REQUIRED_BOOTSTRAP_RESAMPLES + " by the Prelude spec.");
        }

        EvaluationRunEntity existing = em.find(EvaluationRunEntity.class, req.runId());
        if (existing != null) {
            Integer reqPlannedTestN = req.plannedTestN();
            Integer reqPlannedComparisons = req.plannedComparisons();
            if (reqPlannedTestN != null
                    && reqPlannedComparisons != null
                    && existing.getPlannedTestN() == reqPlannedTestN.intValue()
                    && existing.getPlannedComparisons() == reqPlannedComparisons.intValue()
                    && existing.getBootstrapResamples() == resamples
                    && Double.compare(existing.getCiLevel(), ciLevel) == 0) {
                return new RunRegistration.Response(req, true);
            }
            throw new PreludeApiException.ContractViolation("RUN_ALREADY_REGISTERED",
                    "Run '" + req.runId() + "' is already registered with different planning parameters. "
                            + "plannedTestN/plannedComparisons are immutable once registered; use a new runId.");
        }

        Integer plannedTestN = req.plannedTestN();
        Integer plannedComparisons = req.plannedComparisons();
        if (plannedTestN == null || plannedComparisons == null) {
            throw new PreludeApiException.ContractViolation("VALIDATION_ERROR",
                    "plannedTestN and plannedComparisons are required.");
        }
        em.persist(new EvaluationRunEntity(req.runId(), req.description(), req.phase(),
                plannedTestN.intValue(), plannedComparisons.intValue(), resamples, ciLevel,
                req.createdBy(), Instant.now()));
        return new RunRegistration.Response(req, false);
    }

    // --------------------------------------------------------------- capture

    @Transactional
    public IngestionResponse ingestCapture(IngestionEnvelope<CapturePayload> env) {
        requireRun(env.runId());
        String payloadJson = Canonicalizer.write(env.payload());
        String hash = Canonicalizer.sha256Hex(Canonicalizer.fingerprint("capture", env.runId(), payloadJson));
        IngestionResponse duplicate = detectDuplicate(env.submissionId(), hash);
        if (duplicate != null) {
            return duplicate;
        }

        List<CaptureFrameMetricEntity> rows = new ArrayList<>();
        Set<String> imageIds = new TreeSet<>();
        for (CapturePayload.Burst burst : env.payload().bursts()) {
            imageIds.add(burst.imageId());
            // One version per burst so all frames of a burst stay coherent across revisions.
            int version = nextVersion("CaptureFrameMetricEntity",
                    "e.runId = :runId and e.imageId = :imageId",
                    Map.of("runId", env.runId(), "imageId", burst.imageId()));
            Set<Integer> frameIndexes = new HashSet<>();
            for (CapturePayload.Frame f : burst.frames()) {
                if (!frameIndexes.add(f.frameIndex())) {
                    throw new PreludeApiException.ContractViolation("DUPLICATE_FRAME_INDEX",
                            "Duplicate frameIndex " + f.frameIndex() + " in burst " + burst.imageId());
                }
                rows.add(new CaptureFrameMetricEntity(env.runId(), env.submissionId(), burst.imageId(), version,
                        f.frameIndex(), f.iso(), f.exposureTimeNs(), f.capturedAtEpochMs(), f.sharpnessScore(),
                        f.blurRejected(), f.emergencyFallback()));
            }
        }
        persist(env, "capture", payloadJson, hash, imageIds.size(), () -> rows.forEach(em::persist));
        return accepted(env, "capture", imageIds.size(), List.of());
    }

    // ------------------------------------------------------------- alignment

    @Transactional
    public IngestionResponse ingestAlignment(IngestionEnvelope<AlignmentPayload> env) {
        requireRun(env.runId());
        String payloadJson = Canonicalizer.write(env.payload());
        String hash = Canonicalizer.sha256Hex(Canonicalizer.fingerprint("alignment", env.runId(), payloadJson));
        IngestionResponse duplicate = detectDuplicate(env.submissionId(), hash);
        if (duplicate != null) {
            return duplicate;
        }

        List<AlignmentImageMetricEntity> imageRows = new ArrayList<>();
        List<AlignmentFrameMetricEntity> frameRows = new ArrayList<>();
        Set<String> imageIds = new TreeSet<>();
        for (AlignmentPayload.Image img : env.payload().images()) {
            imageIds.add(img.imageId());
            if (img.alignedFrameCount() > img.totalFrameCount()) {
                throw new PreludeApiException.ContractViolation("INCONSISTENT_FRAME_COUNTS",
                        "alignedFrameCount exceeds totalFrameCount for image " + img.imageId());
            }
            int version = nextVersion("AlignmentImageMetricEntity",
                    "e.runId = :runId and e.imageId = :imageId",
                    Map.of("runId", env.runId(), "imageId", img.imageId()));
            imageRows.add(new AlignmentImageMetricEntity(env.runId(), env.submissionId(), img.imageId(), version,
                    img.totalFrameCount(), img.alignedFrameCount(), img.alignmentSuccess(), img.alignmentConfidence(),
                    img.classifierPredictedLabel(), img.classifierGroundTruthLabel(), img.classifierCorrect()));
            Set<Integer> frameIndexes = new HashSet<>();
            for (AlignmentPayload.Frame f : img.frames()) {
                if (!frameIndexes.add(f.frameIndex())) {
                    throw new PreludeApiException.ContractViolation("DUPLICATE_FRAME_INDEX",
                            "Duplicate frameIndex " + f.frameIndex() + " for image " + img.imageId());
                }
                frameRows.add(new AlignmentFrameMetricEntity(env.runId(), env.submissionId(), img.imageId(), version,
                        f.frameIndex(), f.ransacInlierRatio(), f.frameAligned()));
            }
        }

        AlignmentPayload p = env.payload();
        AlignmentClassifierSummaryEntity summary = null;
        if (p.heldOutClassifierAccuracy() != null || p.heldOutClassifierPrecision() != null
                || p.heldOutClassifierRecall() != null) {
            int summaryVersion = nextVersion("AlignmentClassifierSummaryEntity",
                    "e.runId = :runId", Map.of("runId", env.runId()));
            summary = new AlignmentClassifierSummaryEntity(env.runId(), env.submissionId(), summaryVersion,
                    p.heldOutClassifierAccuracy(), p.heldOutClassifierPrecision(), p.heldOutClassifierRecall(),
                    imageIds.size());
        }
        final AlignmentClassifierSummaryEntity summaryRow = summary;
        persist(env, "alignment", payloadJson, hash, imageIds.size(), () -> {
            imageRows.forEach(em::persist);
            frameRows.forEach(em::persist);
            if (summaryRow != null) {
                em.persist(summaryRow);
            }
        });
        return accepted(env, "alignment", imageIds.size(), List.of());
    }

    // ---------------------------------------------------------------- fusion

    @Transactional
    public IngestionResponse ingestFusion(IngestionEnvelope<FusionPayload> env) {
        requireRun(env.runId());
        String payloadJson = Canonicalizer.write(env.payload());
        String hash = Canonicalizer.sha256Hex(Canonicalizer.fingerprint("fusion", env.runId(), payloadJson));
        IngestionResponse duplicate = detectDuplicate(env.submissionId(), hash);
        if (duplicate != null) {
            return duplicate;
        }

        List<String> warnings = new ArrayList<>();
        List<FusionStrategyScoreEntity> rows = new ArrayList<>();
        Set<String> imageIds = new TreeSet<>();
        for (FusionPayload.Image img : env.payload().images()) {
            imageIds.add(img.imageId());
            Set<String> seen = new HashSet<>();
            for (FusionPayload.StrategyScore s : img.scores()) {
                if (!seen.add(s.strategy())) {
                    throw new PreludeApiException.ContractViolation("DUPLICATE_STRATEGY",
                            "Duplicate strategy '" + s.strategy() + "' for image " + img.imageId());
                }
                // Per-(image, strategy) versioning: Phase 3 can supersede only the 'learned' scores
                // without touching the classical strategies' version history.
                int version = nextVersion("FusionStrategyScoreEntity",
                        "e.runId = :runId and e.imageId = :imageId and e.strategy = :strategy",
                        Map.of("runId", env.runId(), "imageId", img.imageId(), "strategy", s.strategy()));
                rows.add(new FusionStrategyScoreEntity(env.runId(), env.submissionId(), img.imageId(), version,
                        s.strategy(), s.ssim(), s.psnr()));
            }
            for (String required : REQUIRED_FUSION_STRATEGIES) {
                if (!seen.contains(required)) {
                    throw new PreludeApiException.ContractViolation("MISSING_REQUIRED_STRATEGY",
                            "Image " + img.imageId() + " is missing required strategy '" + required
                                    + "'. ('learned' is optional until the Phase 3 model ships.)");
                }
            }
            if (!seen.contains("learned")) {
                warnings.add("image " + img.imageId()
                        + ": no 'learned' strategy score (expected until the Phase 3 model ships)");
            }
        }

        FusionModelDiagnosticEntity diagnostic = null;
        if (env.payload().learnedModel() != null) {
            FusionPayload.LearnedModelDiagnostics lm = env.payload().learnedModel();
            double delta = lm.trainSsim() - lm.valSsim();
            if (lm.trainValSsimDelta() != null && Math.abs(lm.trainValSsimDelta() - delta) > 1e-9) {
                warnings.add("trainValSsimDelta recomputed server-side as trainSsim - valSsim (" + delta + ")");
            }
            int diagVersion = nextVersion("FusionModelDiagnosticEntity",
                    "e.runId = :runId and e.modelVersion = :modelVersion",
                    Map.of("runId", env.runId(), "modelVersion", lm.modelVersion()));
            diagnostic = new FusionModelDiagnosticEntity(env.runId(), env.submissionId(), diagVersion,
                    lm.modelVersion(), lm.trainSsim(), lm.valSsim(), delta);
        }
        final FusionModelDiagnosticEntity diagnosticRow = diagnostic;
        persist(env, "fusion", payloadJson, hash, imageIds.size(), () -> {
            rows.forEach(em::persist);
            if (diagnosticRow != null) {
                em.persist(diagnosticRow);
            }
        });
        return accepted(env, "fusion", imageIds.size(), warnings);
    }

    // --------------------------------------------------------------- denoise

    @Transactional
    public IngestionResponse ingestDenoise(IngestionEnvelope<DenoisePayload> env) {
        requireRun(env.runId());
        String payloadJson = Canonicalizer.write(env.payload());
        String hash = Canonicalizer.sha256Hex(Canonicalizer.fingerprint("denoise", env.runId(), payloadJson));
        IngestionResponse duplicate = detectDuplicate(env.submissionId(), hash);
        if (duplicate != null) {
            return duplicate;
        }

        List<String> warnings = new ArrayList<>();
        List<DenoiseImageMetricEntity> rows = new ArrayList<>();
        List<RestormerGoNogoEntity> restormerRows = new ArrayList<>();
        Set<String> imageIds = new TreeSet<>();
        Map<String, Set<String>> precisionByImage = new HashMap<>();

        DenoisePayload p = env.payload();
        for (DenoisePayload.Image img : p.images()) {
            imageIds.add(img.imageId());
            precisionByImage.computeIfAbsent(img.imageId(), k -> new HashSet<>()).add(img.precision());
            int version = nextVersion("DenoiseImageMetricEntity",
                    "e.runId = :runId and e.imageId = :imageId and e.modelVariant = :modelVariant and e.precision = :precision",
                    Map.of("runId", env.runId(), "imageId", img.imageId(),
                            "modelVariant", p.modelVariant(), "precision", img.precision()));
            rows.add(new DenoiseImageMetricEntity(env.runId(), env.submissionId(), img.imageId(), version,
                    p.modelVariant(), img.precision(), img.psnr(), img.ssim(), img.latencyMs(),
                    img.discardRaceEvent(), img.timeoutEvent(), p.modelFileSizeBytes()));
        }
        precisionByImage.forEach((imageId, precisions) -> {
            if (precisions.size() == 1) {
                warnings.add("image " + imageId + ": only '" + precisions.iterator().next()
                        + "' submitted; FP32/INT8 pairs are expected for deployment comparison");
            }
        });

        if (p.restormerComparisons() != null) {
            for (DenoisePayload.RestormerComparison c : p.restormerComparisons()) {
                imageIds.add(c.imageId());
                int version = nextVersion("RestormerGoNogoEntity",
                        "e.runId = :runId and e.imageId = :imageId",
                        Map.of("runId", env.runId(), "imageId", c.imageId()));
                restormerRows.add(new RestormerGoNogoEntity(env.runId(), env.submissionId(), c.imageId(), version,
                        c.baselineInt8Psnr(), c.baselineInt8Ssim(), c.baselineInt8LatencyMs(),
                        c.restormerPsnr(), c.restormerSsim(), c.restormerLatencyMs()));
            }
        }
        persist(env, "denoise", payloadJson, hash, imageIds.size(), () -> {
            rows.forEach(em::persist);
            restormerRows.forEach(em::persist);
        });
        return accepted(env, "denoise", imageIds.size(), warnings);
    }

    // ----------------------------------------------------------- postprocess

    @Transactional
    public IngestionResponse ingestPostProcess(IngestionEnvelope<PostProcessPayload> env) {
        requireRun(env.runId());
        String payloadJson = Canonicalizer.write(env.payload());
        String hash = Canonicalizer.sha256Hex(Canonicalizer.fingerprint("postprocess", env.runId(), payloadJson));
        IngestionResponse duplicate = detectDuplicate(env.submissionId(), hash);
        if (duplicate != null) {
            return duplicate;
        }

        List<String> warnings = new ArrayList<>();
        List<PostProcessImageMetricEntity> rows = new ArrayList<>();
        Set<String> imageIds = new TreeSet<>();
        for (PostProcessPayload.Image img : env.payload().images()) {
            imageIds.add(img.imageId());
            if (img.ssimBeforePostprocess() != null && img.ssimAfterPostprocess() != null) {
                double expected = img.ssimAfterPostprocess() - img.ssimBeforePostprocess();
                if (Math.abs(img.ssimContribution() - expected) > 1e-6) {
                    throw new PreludeApiException.ContractViolation("INCONSISTENT_CONTRIBUTION",
                            "Image " + img.imageId() + ": ssimContribution must equal ssimAfterPostprocess - "
                                    + "ssimBeforePostprocess (expected " + expected + ", got "
                                    + img.ssimContribution() + ")");
                }
            } else {
                warnings.add("image " + img.imageId()
                        + ": ssimBeforePostprocess/ssimAfterPostprocess missing; contribution not cross-checked");
            }
            int version = nextVersion("PostProcessImageMetricEntity",
                    "e.runId = :runId and e.imageId = :imageId",
                    Map.of("runId", env.runId(), "imageId", img.imageId()));
            rows.add(new PostProcessImageMetricEntity(env.runId(), env.submissionId(), img.imageId(), version,
                    img.ssimBeforePostprocess(), img.ssimAfterPostprocess(), img.ssimContribution(),
                    img.fullPipelineSsim()));
        }
        persist(env, "postprocess", payloadJson, hash, imageIds.size(), () -> rows.forEach(em::persist));
        return accepted(env, "postprocess", imageIds.size(), warnings);
    }

    // --------------------------------------------------------------- training

    @Transactional
    public IngestionResponse ingestTraining(IngestionEnvelope<TrainingPayload> env) {
        requireRun(env.runId());
        String payloadJson = Canonicalizer.write(env.payload());
        String hash = Canonicalizer.sha256Hex(Canonicalizer.fingerprint("training", env.runId(), payloadJson));
        IngestionResponse duplicate = detectDuplicate(env.submissionId(), hash);
        if (duplicate != null) {
            return duplicate;
        }

        TrainingPayload p = env.payload();
        List<String> imageIds = p.perImageScores().stream().map(TrainingPayload.ImageScore::imageId).distinct().toList();

        // Leakage guard (contract §8): training/validation submissions may never reference
        // images locked in the held-out test manifest.
        List<String> leaked = em
                .createQuery("select h.imageId from HeldoutTestImageEntity h where h.imageId in :ids", String.class)
                .setParameter("ids", imageIds)
                .getResultList();
        if (!leaked.isEmpty()) {
            throw new PreludeApiException.ContractViolation("TEST_SET_LEAKAGE",
                    "Training/validation submissions may not reference held-out test images. Matched: " + leaked
                            + ". Check candidates against GET /api/v1/testset/manifest before training.");
        }

        List<TrainingImageScoreEntity> scoreRows = new ArrayList<>();
        for (TrainingPayload.ImageScore s : p.perImageScores()) {
            int version = nextVersion("TrainingImageScoreEntity",
                    "e.runId = :runId and e.model = :model and e.epoch = :epoch "
                            + "and e.imageId = :imageId and e.metricName = :metricName",
                    Map.of("runId", env.runId(), "model", p.model(), "epoch", s.epoch(),
                            "imageId", s.imageId(), "metricName", s.metricName()));
            scoreRows.add(new TrainingImageScoreEntity(env.runId(), env.submissionId(), s.imageId(), version,
                    p.model(), s.epoch(), s.globalStep(), s.metricName(), s.value(), s.split()));
        }
        List<TrainingLossPointEntity> lossRows = new ArrayList<>();
        if (p.lossCurves() != null) {
            for (TrainingPayload.LossCurve curve : p.lossCurves()) {
                int version = nextVersion("TrainingLossPointEntity",
                        "e.runId = :runId and e.model = :model and e.curve = :curve",
                        Map.of("runId", env.runId(), "model", p.model(), "curve", curve.curve()));
                for (TrainingPayload.Point point : curve.points()) {
                    lossRows.add(new TrainingLossPointEntity(env.runId(), env.submissionId(), version,
                            p.model(), curve.curve(), point.step(), point.value()));
                }
            }
        }
        persist(env, "training", payloadJson, hash, imageIds.size(), () -> {
            scoreRows.forEach(em::persist);
            lossRows.forEach(em::persist);
        });
        return accepted(env, "training", imageIds.size(), List.of());
    }

    // ---------------------------------------------------------- batch runner

    @Transactional
    public IngestionResponse ingestBatchVariant(IngestionEnvelope<BatchPayload> env) {
        requireRun(env.runId());
        String payloadJson = Canonicalizer.write(env.payload());
        String hash = Canonicalizer.sha256Hex(Canonicalizer.fingerprint("batch_variant", env.runId(), payloadJson));
        IngestionResponse duplicate = detectDuplicate(env.submissionId(), hash);
        if (duplicate != null) {
            return duplicate;
        }

        List<BatchVariantScoreEntity> rows = new ArrayList<>();
        Set<String> imageIds = new TreeSet<>();
        for (BatchPayload.Score s : env.payload().scores()) {
            if (!KNOWN_BATCH_VARIANTS.contains(s.variant())) {
                throw new PreludeApiException.ContractViolation("UNKNOWN_VARIANT",
                        "Unknown evaluation variant '" + s.variant() + "'. Allowed: " + KNOWN_BATCH_VARIANTS
                                + " (docs/data-contract.md §5.7).");
            }
            imageIds.add(s.imageId());
            int version = nextVersion("BatchVariantScoreEntity",
                    "e.runId = :runId and e.imageId = :imageId and e.variant = :variant",
                    Map.of("runId", env.runId(), "imageId", s.imageId(), "variant", s.variant()));
            rows.add(new BatchVariantScoreEntity(env.runId(), env.submissionId(), s.imageId(), version,
                    s.variant(), s.ssim(), s.psnr(), s.latencyMs()));
        }
        persist(env, "batch_variant", payloadJson, hash, imageIds.size(), () -> rows.forEach(em::persist));
        return accepted(env, "batch_variant", imageIds.size(), List.of());
    }

    // ------------------------------------------------------------- bootstrap

    @Transactional
    public IngestionResponse ingestBootstrap(IngestionEnvelope<BootstrapPayload> env) {
        EvaluationRunEntity run = requireRun(env.runId());
        String payloadJson = Canonicalizer.write(env.payload());
        String hash = Canonicalizer.sha256Hex(Canonicalizer.fingerprint("bootstrap", env.runId(), payloadJson));
        IngestionResponse duplicate = detectDuplicate(env.submissionId(), hash);
        if (duplicate != null) {
            return duplicate;
        }

        List<String> warnings = new ArrayList<>();
        List<BootstrapComparisonResultEntity> rows = new ArrayList<>();
        for (BootstrapPayload.Comparison c : env.payload().comparisons()) {
            Integer nImagesBoxed = c.nImages();
            Integer nComparisonsBoxed = c.nComparisons();
            if (nImagesBoxed == null || nComparisonsBoxed == null) {
                throw new PreludeApiException.ContractViolation("VALIDATION_ERROR",
                        "nImages and nComparisons are required for comparison '" + c.comparisonKey() + "'.");
            }
            int nImages = nImagesBoxed.intValue();
            int nComparisons = nComparisonsBoxed.intValue();

            int resamples = c.resamples() == null ? REQUIRED_BOOTSTRAP_RESAMPLES : c.resamples().intValue();
            if (resamples != REQUIRED_BOOTSTRAP_RESAMPLES) {
                throw new PreludeApiException.ContractViolation("UNSUPPORTED_RESAMPLE_COUNT",
                        "Bootstrap resample count is fixed at " + REQUIRED_BOOTSTRAP_RESAMPLES + ".");
            }
            if (Math.abs(c.ciWidth() - (c.ciUpper() - c.ciLower())) > 1e-9) {
                throw new PreludeApiException.ContractViolation("BOOTSTRAP_INCONSISTENT",
                        "ciWidth must equal ciUpper - ciLower for comparison '" + c.comparisonKey() + "'.");
            }
            boolean excludesZero = c.ciLower() > 0 || c.ciUpper() < 0;
            if ("significant".equals(c.significanceStatus()) != excludesZero) {
                throw new PreludeApiException.ContractViolation("BOOTSTRAP_INCONSISTENT",
                        "Comparison '" + c.comparisonKey() + "': significanceStatus must be 'significant' if and "
                                + "only if the Bonferroni-corrected-level CI excludes zero; otherwise report "
                                + "'directional' (docs/data-contract.md §9).");
            }
            if (run.getPlannedTestN() != nImages) {
                warnings.add("comparison " + c.comparisonKey() + ": nImages=" + nImages
                        + " differs from registered plannedTestN=" + run.getPlannedTestN());
            }
            int version = nextVersion("BootstrapComparisonResultEntity",
                    "e.runId = :runId and e.comparisonKey = :comparisonKey and e.metric = :metric",
                    Map.of("runId", env.runId(), "comparisonKey", c.comparisonKey(), "metric", c.metric()));
            rows.add(new BootstrapComparisonResultEntity(env.runId(), env.submissionId(), version,
                    c.comparisonKey(), c.metric(), nImages, nComparisons, 0.05 / nComparisons,
                    c.meanDelta(), c.ciLower(), c.ciUpper(), c.ciWidth(), c.significanceStatus(), resamples,
                    c.computedAt()));
        }
        persist(env, "bootstrap", payloadJson, hash, rows.size(), () -> rows.forEach(em::persist));
        return accepted(env, "bootstrap", rows.size(), warnings);
    }

    // -------------------------------------------------------------- plumbing

    /** Server-assigned supersede version (contract §3.2): strictly monotonic per grouping key —
     *  the target table's uniqueness tuple minus result_version. Never accepted from the caller. */
    private int nextVersion(String entityName, String keyClause, Map<String, Object> keyParams) {
        var query = em.createQuery(
                "select coalesce(max(e.resultVersion), 0) from " + entityName + " e where " + keyClause,
                Number.class);
        keyParams.forEach(query::setParameter);
        Number current = query.getSingleResult();
        return current.intValue() + 1;
    }

    private EvaluationRunEntity requireRun(String runId) {
        EvaluationRunEntity run = em.find(EvaluationRunEntity.class, runId);
        if (run == null) {
            throw new PreludeApiException.RunNotRegistered(runId);
        }
        return run;
    }

    private IngestionResponse detectDuplicate(UUID submissionId, String hash) {
        SubmissionEntity existing = em.find(SubmissionEntity.class, submissionId);
        if (existing == null) {
            return null;
        }
        if (!existing.getPayloadHash().equals(hash)) {
            throw new PreludeApiException.IdempotencyKeyReused(submissionId);
        }
        return new IngestionResponse(existing.getSubmissionId(), existing.getRunId(), existing.getStage(),
                true, existing.getImageCount(), List.of());
    }

    private void persist(IngestionEnvelope<?> env, String stage, String payloadJson, String hash,
                         int imageCount, Runnable inserts) {
        SubmissionEntity submission = new SubmissionEntity(env.submissionId(), env.runId(), stage, env.deviceId(),
                env.appVersion(), env.occurredAt(), Instant.now(), hash, payloadJson, imageCount);
        try {
            em.persist(submission);
            em.flush();
            inserts.run();
            em.flush();
        } catch (PersistenceException pe) {
            // Constraint violations surface at flush; the exception advice classifies them
            // (concurrent duplicate retry vs concurrent supersede) after rollback.
            throw new PreludeApiException.MetricConflictSignal(env.submissionId(), hash, pe);
        }
    }

    private IngestionResponse accepted(IngestionEnvelope<?> env, String stage, int imageCount, List<String> warnings) {
        return new IngestionResponse(env.submissionId(), env.runId(), stage, false, imageCount, warnings);
    }
}