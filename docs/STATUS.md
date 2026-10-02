# Prelude — Build Status (single source of truth)

_Last updated 2026-10-02 by Role 1's **system agent** (verification pass): Read API +
paired bootstrap verified end-to-end, Flyway V1+V2 verified on a fresh volume, and the
repo reconciled with `origin/main` (7 Role 5a commits arrived mid-pass from a stale local
clone — see Resolved history). Prior entry: Role 1, after the first real capture-android
build (JDK 25 / Gradle 9.7.1 / AGP 9.4.0) and the capture-contract live round-trip._

## Resolved history (was broken, now fixed — kept so STATUS stays trustworthy)

- **Flyway never executed on boot.** Root cause: Spring Boot 4 split `FlywayAutoConfiguration`
  into its own `spring-boot-starter-flyway` module — having `flyway-core` +
  `flyway-database-postgresql` on the classpath was not sufficient to register the
  autoconfiguration, so Flyway silently never ran. Fixed by adding `spring-boot-starter-flyway`.
  Verified: migration applies, all 16 tables + `flyway_schema_history` present, app boots,
  `GET /` returns `{"status":"up"}`.
- **Schema type mismatch surfaced once Flyway actually ran.** `submission.payload_hash` and
  `heldout_test_image.sha256` were `CHAR(64)` in the migration but mapped as
  `String(length=64)` (→ `varchar(64)`) in the JPA entities. Fixed in
  `V1__initial_schema.sql` directly — the migration had never been applied, so editing it in
  place was safe.
- **`captureBurst()` replaces sequential locked captures** in capture-android. No technical
  blocker was found (blur rejection is a post-capture step, so nothing requires inspecting a
  frame between captures). Switched to real hardware burst to minimize inter-frame gap, which
  lands directly on Role 2 alignment.
- **The Gradle wrapper jar could not be committed.** The root `.gitignore` rule `*.jar`
  silently blocked `gradle/wrapper/gradle-wrapper.jar`. Fixed with a targeted exception
  (`!**/gradle/wrapper/gradle-wrapper.jar`); wrapper committed pinned to Gradle 9.7.1 with
  `gradlew` mode `100755` (set via `git update-index --chmod`, since Windows cannot represent
  the bit — note `git commit -- <path>` re-reads the working tree and silently resets it).
- **capture-android had never been compiled.** The AGP 9 built-in-Kotlin migration (remove the
  standalone `kotlin-android` plugin, `kotlinOptions` → `kotlin{compilerOptions}`, KGP 2.4.0
  pinned via buildscript classpath — the documented way to raise AGP 9's bundled 2.2.10) plus
  the first real `gradlew build` surfaced four latent defects, all fixed 2026-10-01:
  1. `model/FrameBurst.kt` contained a duplicate of `SharpnessEstimator` instead of the model
     classes — `YuvFrame`/`CapturedFrame`/`FrameBurst` were referenced by 4 files but defined
     nowhere. Reconstructed from usage sites; verified against the data contract (see below).
  2. The `PipelineModeDecider` implementation was lost entirely while its test survived,
     misplaced in the **main** source set. Reimplemented from the test's pinned spec and the
     test moved to `src/test` — 11/11 pass.
  3. `suspendCancellableCoroutine` was imported from `kotlin.coroutines` instead of
     `kotlinx.coroutines`.
  4. `activity_calibration.xml` had a raw `<` inside an `android:hint` attribute (invalid XML).
- **The root `capture-android/build.gradle.kts` was a phantom app module** (applied AGP +
  Kotlin with an `android{}` block but had no sources). Converted to the standard
  `plugins { ... apply false }` + buildscript pin shape.
- **Lint `MissingPermission` on `openCamera` — traced, not suppressed blindly.** There is
  exactly one `openCamera` call site (`BurstCaptureManager.open()`), and both UI paths that
  reach it — `MainActivity` (capture button) and `CalibrationActivity` (`runBlurCalibration`
  focus sweep) — check and request CAMERA permission before constructing the manager. The
  scoped `@SuppressLint` documents that contract.
- **System-agent verification pass (2026-10-02) — three discrepancies found and fixed before
  verification could run:**
  1. `V2__computed_bootstrap_result.sql` contained the `CREATE TABLE` + indexes **twice** —
     Flyway would have failed on the duplicate `CREATE TABLE`. Deduplicated; fresh-volume boot
     then applied V1+V2 cleanly.
  2. `PairedBootstrapTest` existed only as an **empty 5-line stub** in
     `src/test/java/com/prelude/resultsservice/` (wrong package, no tests). The real 11
     property tests were authored into `src/test/java/com/prelude/resultsservice/stats/` per
     the addendum (pairing/order, null handling, input validation, seed determinism,
     independent reference-implementation cross-check, degenerate intervals, Bonferroni
     monotonicity, variant-swap mirror, metadata echo). Note: no ingestion/idempotency/
     leakage-guard JUnit tests exist anywhere in the repo — those service behaviors are
     covered only by the live round-trip and the (still-to-do) Testcontainers CI gate.
  3. **Stale local clone during the pass.** The working tree did **not** contain the Role 5a
     handoff, so an interim `:quantization-deploy` scaffold was applied locally under
     `capture-android/` before the module was found to exist upstream. `git fetch` then
     revealed local `main` was 7 commits behind `origin/main` — including Role 5a's real
     integration (`089a3d5`, `6fd0150`: root-level `quantization-deploy/` Android library
     wired into capture-android via a `projectDir` override, discard-race timeout, tiling,
     `litert = "2.1.5"` pinned — same coordinate `com.google.ai.edge.litert:litert`, plus an
     FP32 conversion smoke test and the nearest-rank p95 calibration API). Resolution: the
     interim scaffold and Gradle edits were **discarded**, `main` fast-forwarded to
     `origin/main` (8ec4bdc), and doc updates were re-applied on top. The post-pull
     `gradlew build` re-verification below covers Role 5a's real module and its tests.

## Reconstructed code — verified against the contract (2026-10-02)

`YuvFrame`, `CapturedFrame`, `FrameBurst` and `PipelineModeDecider` were **reconstructed**
(2026-10-01) from usage sites and the committed test, then confirmed field-by-field against
`docs/data-contract.md` on 2026-10-02:

- §6.1 inter-stage fields match; two property names were corrected to the contract
  (`CapturedFrame.image`, `CapturedFrame.sensorTimestampNs`).
- **Type divergence, flagged:** contract says `CapturedFrame.image: ImageProxy`; the code uses
  `image: YuvFrame` (Camera2 Image buffers are transient, so planes are copied out eagerly;
  `ImageProxy` is CameraX and not available to this Camera2 module). Contract explicitly leaves
  on-device types to the owning role.
- **Semantics divergence, flagged, not silently changed:** the serializer sends
  `capturedAtEpochMs = System.currentTimeMillis()` (wall clock at submit), while §5.1 says
  "Sensor timestamp, epoch ms". A faithful conversion needs the device's
  `SENSOR_INFO_TIMESTAMP_SOURCE` (UNKNOWN=uptime vs REALTIME=elapsedRealtime) captured at
  frame time — a small CaptureModule change that should be a deliberate decision, not a drive-by.
- Automated checks: `ResultsServiceClientTest` validates the envelope against §5.1, and its
  live round-trip test (gated on `PRELUDE_LIVE_URL`; skipped in CI) registered a run and POSTed
  the serializer's own envelope: `POST /api/v1/metrics/capture` → HTTP 200,
  `acceptedImageCount: 1`.

## Newly closed decisions

- **batch-runner architecture** — an **Android instrumented test harness / in-app headless
  runner, not a desktop script.** Reason: it must log per-device thermal state during
  benchmarks via Android's Thermal API (`PowerManager` thermal status, API 30+), which only
  exists on-device — the same reasoning that put the calibration tool inside capture-android.
  Skeleton only; real implementation is blocked on the Roles 2/3/4 pipeline.
- **Calibration tool form factor** — in-app debug mode inside `capture-android` (on-device
  Thermal + TFLite access ruled out a desktop script). Blur/texture threshold implemented;
  the other three calculators are stubs blocked on Roles 2/3/4/5a.
- **Read/Query API (Q8)** — **implemented** (`read/` package: `ReadController`, `ReadService`,
  `ReadDtos`, dedicated exception advice — no existing files edited). Endpoints: paged runs +
  run detail, per-image scores (latest-wins `DISTINCT ON`, `?history=true` for full history),
  ablation (the seven §5.7 variants with n / mean SSIM / PSNR / nearest-rank p95 latency),
  snapshot (canonical demo-app doc incl. SHA-256 `snapshotVersion`), and bootstrap
  compute-and-store. Contract addendum v1.1.0 drafted (`docs/data-contract-addendum-v1.1.0.md`,
  now complete through §13.9; §13 summary appended to the contract). **Pending orchestrator
  ratification.**
- **Paired bootstrap** — pure Java (`stats/` package, no framework imports): 10k resamples,
  seeded `java.util.Random` (JDK-specified sequence → reproducible on any JVM), Bonferroni
  correction (family α 0.05 ÷ run's `plannedComparisons`), `MIN_PAIRED_N = 5`,
  `significant` iff corrected CI excludes zero else `directional`. Covered by 11 property
  tests. Verified live against the smoke run (both metrics `significant`, intervals stored).
- **V2 Flyway migration** — additive `computed_bootstrap_result` table (append-only bootstrap
  storage, server-assigned `result_version`, same §3.2 rule as scores). Verified: fresh volume
  → `Successfully applied 2 migrations`, schema history 2/2 success, 17 app tables.
- **Role 5a integration** — real module integrated by Role 5a on `origin/main` (root-level
  `quantization-deploy/` with denoise/tiling/discard-race sources and tests, wired into
  capture-android via a `projectDir` override; `litert = "2.1.5"` pinned in
  `libs.versions.toml`, mirrored into `docs/versions.md`). Re-verified on this machine
  post-pull: `gradlew clean build` green with the module in the graph.

## Decision log — all formerly open items are resolved

| Item | Resolution |
|---|---|
| Flyway ↔ PostgreSQL 18 | **Verified.** Boot 4.1.0 manages Flyway 12.4.0; PG 18 is a verified supported version for flyway-database-postgresql. Flyway kept; Liquibase fallback retired. |
| Web starter | `spring-boot-starter-webmvc` (Boot 4 rename). No alias reliance. |
| Idempotency/supersede (Q1) | **Ratified as final**, with two locks: `resultVersion` is server-assigned, monotonic per grouping key, never client-supplied; latest-wins (DISTINCT ON) is the documented default for every read path, full history only on explicit request. Implemented. |
| Batch ablation variants (Q2) | Spec's verbatim seven: `single_raw_frame`, `fusion_naive`, `fusion_trimmed`, `fusion_confidence_weighted`, `fusion_learned`, `full_pipeline`, `single_frame_fallback`. Implemented. |
| Fusion train/val delta (Q4) | Confirmed as designed: per-image strategy scores + model-level generalization-gap scalar. |
| Data location (Q5) | Final: `data/{sidd, team-captures, held-out, calibration, checkpoints}/`, gitignored, documented in root README. |
| Manifest admin auth (Q6) | Shared static env token is sufficient. **Intentionally minimal — not a real security boundary.** Do not extend without a new decision. |
| `pipeline_mode` (Q7) | Exact values: `fusion_multi` / `fusion_single` (literal spec values). |
| Read/query API (Q8) | **Implemented 2026-10-02** (see Newly closed decisions). Addendum v1.1.0 drafted — **pending ratification**; reads open on the LAN, writes keep their existing auth. |
| `imageId` convention (Q9) | Confirmed: one ID assigned at capture (FrameBurst `burstId`), carried unchanged through every stage and submission. |

## Module status

| Module | Owner | Status |
|---|---|---|
| Data contract (`docs/data-contract.md`) | 1 | ✅ v1.0.0 ratified |
| Results Service ingestion API | 1 | ✅ Implemented incl. server-assigned supersede versions; capture stage verified by live client round-trip (2026-10-02) |
| Results Service schema evolution | 1 | ✅ Flyway + `ddl-auto: validate`; add-only policy in force |
| Read/query API (demo app, bootstrap retrieval) | 1 | ✅ Implemented (pending ratification of addendum v1.1.0); verified live incl. bootstrap compute-and-store (2026-10-02) |
| Capture (Android) | 1 | ✅ Builds + unit tests green on JDK 25 / Gradle 9.7.1 / AGP 9.4.0 built-in Kotlin (2026-10-02). Camera path is static-review only — no device run yet. |
| Calibration tool | 1 | In-app debug mode (closed decision). Blur/texture threshold built; alignment floor + min-frame-count + inference timeout stubbed, blocked on Roles 2/3/4/5a. |
| Pipeline-mode orchestration (`fusion_multi`/`fusion_single`) | 1 | ✅ Pure decider (`com.prelude.pipeline.PipelineModeDecider`) reimplemented from its test — 11/11 unit tests. |
| Batch runner | 1 | Architecture decided (on-device instrumented harness). Skeleton only — blocked on full pipeline. |
| Alignment + confidence classifier | 2 | Not started |
| Fusion (4-way) + post-process + demo UI | 3 | Phase 1 demo code in `fusion-postprocess-demo/` (strategies, tone mapping, sharpening, SSIM search — unit tested); on-device pipeline pending Role 2 |
| Primary CNN training | 4 | Not started |
| Quantization + deployment + discard-race | 5a | ✅ Discard-race timeout + Android module scaffold implemented. Conversion & parity gate proven (litert-torch FP32 pass). LiteRT inference blocked by Role 4 checkpoint (2026-10-02) |
| Restormer stretch | 5b | Unblocked for Phase 1–2 smoke test |

## Open issues & architectural gaps (escalated to the orchestrator — do NOT silently fix)

- **`alignmentConfidenceFloor` delivery gap.** There is **no runtime delivery path and no API
  format** for the calibrated alignment-confidence floor: the calculator in the calibration
  tool is a stub blocked on Role 2's RANSAC, and `CalibrationRepository` neither persists nor
  loads the value. **Decision needed from the orchestrator:** on-device read path vs. a
  Results Service endpoint. **Do not invent a wire format until that decision is made.**
- **Role 2 RANSAC labeling threshold.** Role 2 asked for a calibrated inlier-ratio cutoff for
  training labels; that is a **training-time artifact, not the runtime floor** (distinct from
  the gap above). Recommended approach: data-driven — derive the cutoff from real RANSAC
  output once Role 2 builds it, rather than picking a constant now.

## Engineering notes

- Testcontainers smoke test (migrate + happy path + conflict path) stays as a CI gate — standard regression coverage, not a live unknown.
- Manifest admin auth is intentionally minimal (Q6); treat it as a convenience lock, not a security boundary.
- The Results Service contract round-trip test (`ResultsServiceClientTest`) skips itself
  unless `PRELUDE_LIVE_URL` is set — CI runs it as a static schema check only.
