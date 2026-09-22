# Prelude — Build Status (single source of truth)

_Last updated by Role 1 after orchestrator ratification of the data contract (v1.0.0)._

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

## Newly closed decisions

- **batch-runner architecture** — an **Android instrumented test harness / in-app headless
  runner, not a desktop script.** Reason: it must log per-device thermal state during
  benchmarks via Android's Thermal API (`PowerManager` thermal status, API 30+), which only
  exists on-device — the same reasoning that put the calibration tool inside capture-android.
  Skeleton only; real implementation is blocked on the Roles 2/3/4 pipeline.

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
| Read/query API (Q8) | **Out of scope — flagged as the next brief** (demo-app reads, bootstrap report retrieval). |
| `imageId` convention (Q9) | Confirmed: one ID assigned at capture (FrameBurst `burstId`), carried unchanged through every stage and submission. |

## Module status

| Module | Owner | Status |
|---|---|---|
| Data contract (`docs/data-contract.md`) | 1 | ✅ v1.0.0 ratified |
| Results Service ingestion API | 1 | ✅ Implemented incl. server-assigned supersede versions |
| Results Service schema evolution | 1 | ✅ Flyway + `ddl-auto: validate`; add-only policy in force |
| Read/query API (demo app, bootstrap retrieval) | 1 | ⏭ Next brief (Q8) |
| Capture (Android) | 1 | Not started (unblocked by ratified contract) |
| Calibration tool | 1 | Not started (form-factor decision belongs to a separate brief) |
| Batch runner | 1 | Not started — blocked on working pipeline |
| Alignment + confidence classifier | 2 | Not started |
| Fusion (4-way) + post-process + demo UI | 3 | Not started (blocked by Role 2 output) |
| Primary CNN training | 4 | Not started |
| Quantization + deployment + discard-race | 5a | Blocked by Role 4 model |
| Restormer stretch | 5b | Unblocked for Phase 1–2 smoke test |
| Pipeline-mode orchestration (`fusion_multi`/`fusion_single`) | 1 | Built + unit-tested as a pure decider (`com.prelude.pipeline.PipelineModeDecider`). |
| batch-runner | 1 | Architecture decided (on-device instrumented harness). Skeleton only — blocked on full pipeline. |

## Engineering notes

- Testcontainers smoke test (migrate + happy path + conflict path) stays as a CI gate — standard regression coverage, not a live unknown.
- Manifest admin auth is intentionally minimal (Q6); treat it as a convenience lock, not a security boundary.