# Prelude Data Contract — Results Service Ingestion & Pipeline Interchange
- **Owner:** Role 1
- **Status:** v1.0.0 — **ratified** by the orchestrator
- **Amendments locked at ratification:**
  1. `resultVersion` is server-assigned, monotonic per grouping key — never client-supplied (§3.2).
  2. Latest-wins is the documented default for **every** read path; full history only on explicit request (§3.2, §11).
  3. §5.7 variant list replaced with the spec's verbatim seven.
  4. `pipeline_mode` values final: `fusion_multi` / `fusion_single` (§6.6).
  5. `imageId` convention confirmed (§2.3).

## 1. Purpose & scope

This document is the single authoritative contract for:

1. The metric payloads every pipeline stage submits to the Results Service.
2. The idempotency and supersede semantics of ingestion.
3. The pipeline-internal object contracts handed between modules (FrameBurst → AlignedFrameSet → FusedFrame → DenoisedFrame → FinalImage).
4. Schema evolution policy for the Results Service database.
5. The held-out test-set leakage-prevention mechanism.
6. Bootstrap significance reporting requirements.

The Results Service **ingests metrics only**. It never receives image bytes. Images are referenced exclusively by `imageId` (and, for the held-out manifest, SHA-256 content hashes).

---

## 2. Submission model

### 2.1 Envelope

Every ingestion request uses the same envelope; only `payload` varies by stage.

| Field | Type | Required | Notes |
|---|---|---|---|
| `submissionId` | UUID | yes | **The idempotency key.** Client-generated, fresh per logical submission, reused only when retrying the *exact same* submission. See §3. |
| `runId` | string | yes | Must be a registered evaluation run (§4). |
| `deviceId` | string | no | Capturing/evaluating device. |
| `appVersion` | string | no | Client build. |
| `occurredAt` | ISO-8601 instant | no | When the measurement happened. |
| `payload` | object | yes | Stage-specific; sections §5.1–§5.8. |

`resultVersion` was removed from the envelope at ratification: supersede versions are assigned by the server (§3.2). If a client still sends the field, it is ignored.

**Idempotency fingerprint.** Server-side, retries are detected by hashing `(stage, runId, payload)` — `occurredAt`, `deviceId`, `appVersion` are metadata and may legitimately differ between retries. `submissionId` must be held stable across retries.

**Idempotency fingerprint.** Server-side, retries are detected by hashing `(stage, runId, resultVersion, payload)` — `occurredAt`, `deviceId`, `appVersion` are metadata and may legitimately differ between retries. `submissionId` must be held stable across retries.

### 2.2 Stage names

`capture`, `alignment`, `fusion`, `denoise`, `postprocess`, `training`, `batch_variant`, `bootstrap`.

### 2.3 `imageId` convention (confirmed)

A single identifier is assigned **once by CaptureModule** (as the FrameBurst's `burstId`) and is carried **unchanged** through every downstream stage and every metric submission that references that image. On the wire it is `imageId`; on-device it is `burstId` — the same value. Without this, per-image scores from different pipeline variants cannot be paired for bootstrap significance checks, which is the point of the per-image mandate. Recommended format: `<deviceId>:<burstStartTimestampNs>`.

### 2.4 Per-image mandate

**Aggregate-only submissions are rejected.** Every stage payload must carry per-image (or per-frame, for capture) rows; the service validates this. Aggregates may be included *in addition* (e.g., the alignment classifier summary) but never instead. Rationale: bootstrap resampling (§9) requires per-image granularity; any aggregate that cannot be recomputed from stored per-image rows is not auditable.

---

## 3. Idempotency & supersede semantics ⚠ RATIFY

The Prelude spec requires both **safe retries** and **legitimate score revision** (Phase 3's learned fusion model refines Phase 2's fusion-ablation numbers). A naive `(image_id, stage)` idempotency key would swallow the revision as a duplicate. This contract resolves the conflict with two separate mechanisms:

### 3.1 Transport-level idempotency (safe retries)

- `submissionId` (client-generated UUID) is the idempotency key and the primary key of the `submission` table.
- **Same `submissionId`, same payload** → the original response is replayed (`"duplicate": true`, HTTP 200). Nothing is re-inserted. Retries are always safe.
- **Same `submissionId`, different payload** → HTTP 409 `IDEMPOTENCY_KEY_REUSED`. Keys are never recycled for new content.
- Conflicts are enforced by database constraints, so concurrent identical retries are also safe (exactly-once insertion).

### 3.2 Domain-level supersede (score revision) — server-assigned versions

- Every per-image row carries `(runId, imageId, [discriminator], resultVersion)`, with a uniqueness constraint on that tuple. The discriminator is stage-specific (e.g., `frame_index` for capture, `strategy` for fusion, `model_variant`+`precision` for denoise, `variant` for batch). The tuple minus `resultVersion` is the **grouping key**.
- **`resultVersion` is assigned by the server, never by the caller.** On acceptance, each row receives `MAX(resultVersion) + 1` over its grouping key. Versions are strictly monotonic per key, with no collisions and no gaps (a rolled-back submission consumes no number). Client-supplied version numbers were explicitly rejected at ratification: they invite collisions and gaps across retrying clients. If a client still includes `resultVersion` in the envelope, it is ignored.
- **To revise a result** (e.g., Phase 3 re-scoring Phase 2 fusion numbers with the learned model), submit again with a **fresh `submissionId`**. The server assigns the next version automatically and the new rows become latest. Nothing is bumped by hand and nothing is ever overwritten or deleted. Fusion scores are versioned per `(image, strategy)`, so Phase 3 can supersede only the `learned` scores without touching the classical strategies' history.
- **Latest version wins — on every read path.** All queries (batch runner, analysis, any future read API) must default to `MAX(resultVersion)` per grouping key (canonical SQL in §11). Full history is returned only when a caller explicitly asks for it. This default is normative, not a suggestion.
- Concurrent submissions racing on the same grouping key: one wins; the loser receives `409 CONCURRENT_SUPERSEDE`, which is **retryable** — the retry receives the next version.

### 3.3 Why append-only versioning (and not overwrite)

Overwriting destroys audit trail and makes bootstrap resampling non-reproducible (a resample run could observe half-old/half-new data mid-revision). Append-only versioning keeps every historical value, lets analyses pin a version, and makes "who changed what" answerable from `submission` rows.

---

## 4. Evaluation runs & planned test-set N

Before any submission, the campaign must be registered:

`POST /api/v1/runs`

| Field | Type | Required | Notes |
|---|---|---|---|
| `runId` | string | yes | Stable human-readable ID, e.g. `phase2-fusion-ablation`. |
| `description`, `phase`, `createdBy` | string | no | |
| `plannedTestN` | int > 0 | **yes** | Locked at registration. Bootstrap significance (§9) is judged against this N. |
| `plannedComparisons` | int > 0 | **yes** | Bonferroni family size for this run; fixes the corrected α. |
| `bootstrapResamples` | int | no | Must be 10000 (spec-fixed); default 10000. |
| `ciLevel` | double | no | Default 0.95. |

Registration is **immutable**: re-registering the same `runId` with identical planning parameters is an idempotent no-op; with different `plannedTestN`/`plannedComparisons` it is rejected (`RUN_ALREADY_REGISTERED`). Changing N mid-campaign would invalidate significance claims — start a new run instead. Submissions referencing an unregistered run are rejected (`RUN_NOT_REGISTERED`).

---

## 5. Stage metric schemas

Types: `int`, `long`, `double`, `bool`, `string`, `instant`. Optional fields may be omitted or null.

### 5.1 `capture` (Role 1)

`payload.bursts[]` — one entry per captured burst (`imageId` = burst ID):

| Field | Type | Required | Notes |
|---|---|---|---|
| `imageId` | string | yes | Burst ID; see §2.3. |
| `frames[]` | list | yes (≥1) | Per-frame rows. |
| `frames[].frameIndex` | int ≥ 0 | yes | Unique within a burst. |
| `frames[].iso` | int | yes | Sensor ISO. |
| `frames[].exposureTimeNs` | long | yes | |
| `frames[].capturedAtEpochMs` | long | yes | Sensor timestamp, epoch ms. |
| `frames[].sharpnessScore` | double | yes | On-device sharpness (Laplacian variance or equivalent; method is CaptureModule's call). |
| `frames[].blurRejected` | bool | yes | Frame rejected as blurred. |
| `frames[].emergencyFallback` | bool | yes | Frame captured under emergency-fallback mode. |

### 5.2 `alignment` (Role 2)

`payload.images[]`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `imageId` | string | yes | |
| `totalFrameCount` / `alignedFrameCount` | int | yes | `aligned ≤ total` enforced. |
| `alignmentSuccess` | bool | yes | |
| `alignmentConfidence` | double 0..1 | no | Learned classifier's confidence — recorded for evaluation & orchestration only; **fusion must never consume it** (§6). |
| `classifierPredictedLabel` / `classifierGroundTruthLabel` | string | no | Per-image classifier evaluation against the held-out set. |
| `classifierCorrect` | bool | no | |
| `frames[]` | list | yes (≥1) | |
| `frames[].frameIndex` | int ≥ 0 | yes | |
| `frames[].ransacInlierRatio` | double | yes | **The confidence signal fusion is allowed to use.** |
| `frames[].frameAligned` | bool | yes | |

Optional run-level aggregates (allowed *in addition* to the per-image rows): `heldOutClassifierAccuracy`, `heldOutClassifierPrecision`, `heldOutClassifierRecall` (doubles). The **confidence distribution** for a run is derived by aggregating `alignmentConfidence` across images server-side; no separate histogram field is submitted.

### 5.3 `fusion` (Role 3)

`payload.images[]`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `imageId` | string | yes | |
| `scores[]` | list | yes (≥1) | One row per strategy evaluated. |
| `scores[].strategy` | enum | yes | `naive` \| `trimmed` \| `confidence_weighted` \| `learned`. |
| `scores[].ssim` | double | yes | Per-image fusion score vs. ground truth. |
| `scores[].psnr` | double | no | |

**Coverage rule:** `naive`, `trimmed`, `confidence_weighted` are required for every image; `learned` is optional until the Phase 3 model ships (a warning is returned when it is absent). Duplicate strategies per image are rejected.

Optional `payload.learnedModel` (model-level overfit diagnostic — inherently aggregate; see Q4):

| Field | Type | Required | Notes |
|---|---|---|---|
| `modelVersion` | string | yes | |
| `trainSsim` / `valSsim` | double | yes | |
| `trainValSsimDelta` | double | no | **Recomputed server-side as `trainSsim - valSsim`**; submitted values that disagree produce a warning. |

### 5.4 `denoise` (Roles 5a / 5b)

Payload-level: `modelVariant` (string, required, e.g. `cnn-v1`, `restormer-ft`), `modelFileSizeBytes` (long, optional).

`payload.images[]` — one row per (image, precision):

| Field | Type | Required | Notes |
|---|---|---|---|
| `imageId` | string | yes | |
| `precision` | enum | yes | `fp32` \| `int8`. |
| `psnr` / `ssim` | double | yes | Per-image, vs. ground truth. |
| `latencyMs` | double | yes | Per-image inference latency. **Run-level p95 is derived from these rows** (p95 = nearest-rank 95th percentile); it is not submitted directly. |
| `discardRaceEvent` | bool | yes | Discard-race triggered (Role 5a). |
| `timeoutEvent` | bool | yes | Denoise timeout hit. Timeout budget is owned by Role 5a and is **not** governed by `pipeline_mode`. |

`payload.restormerComparisons[]` (Role 5b go/no-go, anchored to Role 5a's INT8 numbers):

| Field | Type | Required | Notes |
|---|---|---|---|
| `imageId` | string | yes | |
| `baselineInt8Psnr` / `baselineInt8Ssim` | double | **yes** | **Anchoring is enforced by schema**: a Restormer comparison without Role 5a's INT8 baseline numbers is invalid. |
| `baselineInt8LatencyMs` | double | no | |
| `restormerPsnr` / `restormerSsim` | double | yes | |
| `restormerLatencyMs` | double | no | |

A warning is emitted when an image has only one of the two precisions (FP32/INT8 pairs are expected for deployment comparison).

### 5.5 `postprocess` (Role 3)

`payload.images[]`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `imageId` | string | yes | |
| `ssimBeforePostprocess` / `ssimAfterPostprocess` | double | no | If both present, `ssimContribution = after − before` is enforced (tolerance 1e-6) or the submission is rejected (`INCONSISTENT_CONTRIBUTION`). |
| `ssimContribution` | double | yes | **Post-processing's isolated SSIM contribution.** |
| `fullPipelineSsim` | double | no | The combined full-pipeline number, reported **separately** — never conflated with the contribution. |

### 5.6 `training` (Roles 2, 3, 4, 5b)

Payload-level: `model` (string, required — e.g. `denoise_cnn`, `fusion_net`, `alignment_classifier`, `restormer_ft`).

`payload.perImageScores[]` — **must be non-empty** (per-image mandate):

| Field | Type | Required | Notes |
|---|---|---|---|
| `imageId` | string | yes | |
| `epoch` | int ≥ 0 | yes | |
| `globalStep` | int | no | |
| `metricName` | string | yes | `psnr`, `ssim`, `accuracy`, … |
| `value` | double | yes | |
| `split` | enum | yes | `train` \| `validation`. **`heldout_test` does not exist here by design** — see §8: the ingestion endpoint itself rejects training/validation submissions that reference held-out images (`TEST_SET_LEAKAGE`). |

`payload.lossCurves[]` (optional aggregates): `curve` (string, e.g. `train_loss`, `val_loss`), `points[]` of `{step, value}`.

### 5.7 `batch_variant` (Role 1 batch runner)

`payload.scores[]` — per-image results for each of the 7 evaluation variants, verbatim from the spec's evaluation table:

| # | `variant` | Definition (spec) |
|---|---|---|
| 1 | `single_raw_frame` | Single raw frame (no processing) — absolute baseline |
| 2 | `fusion_naive` | Naive multi-frame average (no AI) |
| 3 | `fusion_trimmed` | Trimmed-mean fusion (no AI) |
| 4 | `fusion_confidence_weighted` | Confidence-weighted fusion (no AI) |
| 5 | `fusion_learned` | Learned fusion weighting (no AI) |
| 6 | `full_pipeline` | Full pipeline (best fusion strategy + AI denoising + post-processing) |
| 7 | `single_frame_fallback` | Single-frame fallback (best available frame + AI denoising + post-processing, no fusion) |

| Field | Type | Required | Notes |
|---|---|---|---|
| `imageId` | string | yes | |
| `variant` | enum | yes | Values above; unknown values rejected (`UNKNOWN_VARIANT`). |
| `ssim` | double | yes | |
| `psnr` | double | no | |
| `latencyMs` | double | no | End-to-end for that variant. |

### 5.8 `bootstrap` (whoever computes the comparison; expected: batch runner / analysis)

`payload.comparisons[]`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `comparisonKey` | string | yes | e.g. `full_pipeline__vs__fusion_plus_denoise`. |
| `metric` | string | yes | `ssim`, `psnr`, `latency_ms`, … |
| `nImages` | int > 0 | yes | Test-set N actually used; cross-checked against the run's `plannedTestN` (warning on mismatch). |
| `nComparisons` | int > 0 | yes | Bonferroni family size. |
| `meanDelta` | double | yes | Paired mean difference. |
| `ciLower` / `ciUpper` / `ciWidth` | double | yes | CI at the **corrected** confidence level; `ciWidth = ciUpper − ciLower` enforced. |
| `significanceStatus` | enum | yes | `significant` \| `directional` — see §9. |
| `resamples` | int | no | Must be 10000 if present. |
| `computedAt` | instant | yes | |

Corrected α is stored server-side as `0.05 / nComparisons`.

---

## 6. Inter-stage pipeline object contracts (on-device)

Fixed I/O per the module table; any module may be replaced/benchmarked in isolation as long as it honors these shapes. Field names below are the canonical contract; the Android-side types live with their owning roles.

### 6.1 `FrameBurst` — CaptureModule → AlignmentModule

| Field | Type | Notes |
|---|---|---|
| `burstId` | string | Becomes `imageId` for all downstream reporting (§2.3). |
| `frames` | `List<CapturedFrame>` | |
| `CapturedFrame.frameIndex` | int | |
| `CapturedFrame.image` | `ImageProxy` | Format (YUV_420_888 vs RAW_SENSOR) is a capture-config decision but **must be uniform within a burst**. |
| `CapturedFrame.sensorTimestampNs` | long | |
| `CapturedFrame.iso` | int | |
| `CapturedFrame.exposureTimeNs` | long | |
| `CapturedFrame.sharpnessScore` | float | |
| `CapturedFrame.blurRejected` | bool | |
| `emergencyFallbackActive` | bool | Burst-level fallback flag. |

### 6.2 `AlignedFrameSet` — AlignmentModule → FusionModule

| Field | Type | Notes |
|---|---|---|
| `burstId` | string | |
| `referenceFrameIndex` | int | |
| `frames` | `List<AlignedFrame>` | |
| `AlignedFrame.frameIndex` | int | |
| `AlignedFrame.image` | `ImageProxy` | |
| `AlignedFrame.homography` | `float[9]` | Warp into reference-frame coordinates. |
| `AlignedFrame.ransacInlierRatio` | float | **The only confidence FusionModule may condition on.** |
| `AlignedFrame.aligned` | bool | |
| `alignmentConfidence` | float | Learned classifier output. Diagnostic + orchestration input only. |
| `pipelineMode` | enum | Set by orchestration from alignment confidence; see §6.6. |

**Invariant:** FusionModule consumes `ransacInlierRatio`, **never** the classifier's prediction/confidence. Violations invalidate cross-strategy comparisons.

### 6.3 `FusedFrame` — FusionModule → DenoiseModule

| Field | Type | Notes |
|---|---|---|
| `burstId` | string | |
| `strategy` | enum | Which strategy produced the handed-forward frame. |
| `image` | float buffer | Linear-domain fused frame. |
| `evaluationSidecars` | map strategy → fused candidate | Present in evaluation mode only, so the batch runner can score all 4 strategies; production path carries only `strategy`'s output. |
| `pipelineMode` | enum | Passed through. |

### 6.4 `DenoisedFrame` — DenoiseModule → PostProcessModule

| Field | Type | Notes |
|---|---|---|
| `burstId` | string | |
| `image` | float buffer | |
| `modelVariant` | string | e.g. `cnn-v1`, `restormer-ft`. Runtime is LiteRT 2.1.5 (TF 2.20 lock deprecates `tf.lite`). |
| `precision` | enum | `fp32` \| `int8`. |
| `latencyNs` | long | |
| `discardRaceOccurred` / `timeoutOccurred` | bool | Timeout budget owned by Role 5a; **`pipelineMode` must not alter it**. |

### 6.5 `FinalImage` — PostProcessModule → output

| Field | Type | Notes |
|---|---|---|
| `burstId` | string | |
| `image` | sRGB buffer | Final deliverable. |
| `ssimBeforePostprocess` / `ssimAfterPostprocess` | float? | For the §5.5 contribution split. |
| `pipelineMode` | enum | Recorded for provenance. |

### 6.6 `pipeline_mode`

Orchestration flag driven **only by alignment confidence**, governing **alignment → fusion only** (it must not touch the denoise timeout). Exact values (literal spec values):

- `fusion_multi` — full-burst multi-frame fusion path.
- `fusion_single` — single-frame fallback path (mirrors evaluation variant `single_frame_fallback`).

---

## 7. Schema management

- **Tool: Flyway** (plain-SQL migrations, `results-service/src/main/resources/db/migration`). Choice rationale: SQL-first migrations make the add-only policy trivially reviewable in diffs; migrations are versioned artifacts that map 1:1 to team votes; first-class Spring Boot auto-configuration. (Liquibase is the documented fallback — see open questions — if the Spring-Boot-managed Flyway version does not recognize PostgreSQL 18.)
- **Hibernate `ddl-auto: validate`.** The previous `update` setting is removed: `update` can alter/drop columns from entity drift and would silently violate the evolution policy. `validate` fails startup on entity↔schema mismatch instead.
- **Post-freeze evolution is add-only.** Allowed: `ADD COLUMN` (nullable or defaulted), `CREATE TABLE`, `CREATE INDEX`. Forbidden without a superseding contract version: `DROP`, type changes, renames, constraint tightening. Every change ships as a new `V{n}__description.sql` migration + team vote recorded in `docs/STATUS.md`. Enums are enforced at the application layer (not DB `CHECK` constraints) precisely so value-set growth stays add-only.
- Contract document changes follow the same process: additive field additions may be proposed by any role; semantic changes need orchestrator sign-off.

---

## 8. Test-set leakage prevention

The rule ("never train/validate on the held-out test set") is enforced by three mechanisms, not just policy:

1. **Locked manifest.** Table `heldout_test_image(image_id, sha256, source, manifest_version)` seeded by Role 1 once the held-out set is frozen. Rows are only ever **added** (versioned batches); the API exposes no update or delete path. Mutation requires the shared admin token (`X-Prelude-Admin-Token`). This is **intentionally minimal** — a shared static token via environment variable — because the service runs on a local team machine on local Wi-Fi, not as a public service. It is **not** a production security boundary and must not be mistaken for one.
2. **Programmatic check** for the four independent training pipelines (Roles 2, 3, 4, 5b):
   - `GET /api/v1/testset/manifest` — download the full list of `(imageId, sha256)` for build-time exclusion filters.
   - `POST /api/v1/testset/check` — submit candidate `{imageId?, sha256?}` entries; response partitions them into `heldout` vs `notInHeldout`. Absence from the manifest means "not in the held-out set" — the manifest is the complete locked list.
   - Training pipelines should call `/check` (or diff the manifest) as a pre-training CI step.
3. **Ingestion-time guard.** The `training` endpoint cross-checks every submitted training/validation `imageId` against the manifest and rejects the whole submission with `TEST_SET_LEAKAGE` on any match. A leaked result can therefore never enter the Results Service.

---

## 9. Bootstrap significance reporting

Fixed parameters (spec): **10,000 paired resamples, 95% CI family, Bonferroni correction** across the `nComparisons` comparisons in the run.

- **Test-set N is decided up front**: `plannedTestN` is locked at run registration (§4) and every bootstrap submission reports the `nImages` actually used.
- CIs must be constructed at the **corrected** confidence level (Bonferroni-adjusted), so the decision rule is: CI excludes zero ⇔ significant. The service enforces this equivalence: `significanceStatus = "significant"` is rejected when the CI contains zero (`BOOTSTRAP_INCONSISTENT`).
- **`significanceStatus`:**
  - `significant` — corrected CI excludes zero.
  - `directional` — CI still contains zero after correction (including the case where all planned test images were used and the CI is still too wide). Results at this level must be reported as directional only, everywhere.

---

## 10. REST API reference

Base URL: `/api/v1` (plus the legacy root health endpoint `GET /` → `{"status":"up"}`).

| Method & path | Purpose |
|---|---|
| `POST /api/v1/runs` | Register an evaluation run (idempotent; immutable planning fields). |
| `POST /api/v1/metrics/capture` | §5.1 |
| `POST /api/v1/metrics/alignment` | §5.2 |
| `POST /api/v1/metrics/fusion` | §5.3 |
| `POST /api/v1/metrics/denoise` | §5.4 |
| `POST /api/v1/metrics/postprocess` | §5.5 |
| `POST /api/v1/metrics/training` | §5.6 |
| `POST /api/v1/metrics/batch-variant` | §5.7 |
| `POST /api/v1/metrics/bootstrap` | §5.8 |
| `GET /api/v1/testset/manifest` | Held-out manifest (§8). |
| `POST /api/v1/testset/check` | Leakage check (§8). |
| `POST /api/v1/testset/images` | Admin: append manifest rows (token-guarded). |

**Success:** `201` on first acceptance; `200` with `"duplicate": true` on idempotent replay. Body: `{submissionId, runId, stage, duplicate, acceptedImageCount, warnings[]}`.

**Errors** (body `{code, message, details?}`):

| Status | Codes |
|---|---|
| 400 | `VALIDATION_ERROR`, `MALFORMED_REQUEST`, `RUN_NOT_REGISTERED`, `AGGREGATE_ONLY_REJECTED`*`, `TEST_SET_LEAKAGE`, `BOOTSTRAP_INCONSISTENT`, `INCONSISTENT_CONTRIBUTION`, `MISSING_REQUIRED_STRATEGY`, `DUPLICATE_STRATEGY`, `DUPLICATE_FRAME_INDEX`, `INCONSISTENT_FRAME_COUNTS`, `UNKNOWN_VARIANT`, `UNSUPPORTED_RESAMPLE_COUNT`, `RUN_ALREADY_REGISTERED`, `INCOMPLETE_CANDIDATE`, `MANIFEST_CONFLICT` |
| 401/403 | `ADMIN_AUTH_FAILED` |
| 409 | `IDEMPOTENCY_KEY_REUSED`, `CONCURRENT_SUPERSEDE` (retryable) |

\* `AGGREGATE_ONLY_REJECTED` is expressed structurally: payloads whose per-image lists are empty/missing fail validation with this code in the message.

### Example (capture)

```json
{
  "submissionId": "3f7c1a2e-9b4d-4c1e-8a2f-1c5d9e0b7a31",
  "runId": "phase2-capture-baseline",
  "deviceId": "pixel-7a-01",
  "appVersion": "0.4.0",
  "occurredAt": "2026-01-12T19:42:07.123Z",
  "payload": {
    "bursts": [{
      "imageId": "pixel-7a-01:1768246927123000000",
      "frames": [
        {"frameIndex": 0, "iso": 3200, "exposureTimeNs": 83000000, "capturedAtEpochMs": 1768246927123, "sharpnessScore": 0.42, "blurRejected": false, "emergencyFallback": false},
        {"frameIndex": 1, "iso": 3200, "exposureTimeNs": 83000000, "capturedAtEpochMs": 1768246927241, "sharpnessScore": 0.11, "blurRejected": true,  "emergencyFallback": false}
      ]
    }]
  }
}
```

### Supersede example (Phase 3 revising Phase 2 fusion numbers)

Same `runId` and `imageId`, same strategies, a fresh `submissionId` — nothing else. The server assigns `resultVersion = 2` to the new rows; both versions remain stored; all read paths return version 2 by default (§3.2).

## 11. Querying latest versions (canonical pattern)
Latest-wins is the **default for every read path** (§3.2). Full history is returned only when a caller explicitly asks for it — do not leave that implicit in SQL.

```sql
-- Latest per-image fusion scores for a run:
SELECT DISTINCT ON (image_id, strategy) *
FROM fusion_strategy_score
WHERE run_id = 'phase3-learned-fusion-v1'
ORDER BY image_id, strategy, result_version DESC;

-- p95 latency for a deployment run (spec comparison point):
SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
FROM denoise_image_metric
WHERE run_id = 'phase3-int8-deploy' AND precision = 'int8';
```

---

## 12. Compatibility & verification status

- **Verified:** Spring Boot 4.1.0 manages **Flyway 12.4.0** via spring-boot-dependencies, and `flyway-database-postgresql` lists **PostgreSQL 18** among its verified supported versions. No compatibility gap — Flyway is final; the Liquibase fallback is retired.
- Web starter is **`spring-boot-starter-webmvc`** (Spring Boot 4 modularization renamed `spring-boot-starter-web`; no reliance on any compatibility alias).
- Transitive versions remain inherited from the Boot 4.1.0 parent; nothing pinned in `pom.xml`.
- The Testcontainers smoke test (migrate + happy path + conflict path) stays as a CI gate as **standard regression coverage**, no longer as a compatibility probe.