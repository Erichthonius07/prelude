## 13. Read & query API (addendum v1.1.0)

This addendum is **add-only**. Every §10 ingestion endpoint keeps its current meaning.
It defines the read/query surface of the Results Service plus the server-computed paired
bootstrap.

### 13.1 Authentication posture
- **Writes keep their existing auth.** Held-out manifest mutation (§8) stays guarded by the
  shared admin token (`X-Prelude-Admin-Token`). Ingestion (§10) stays as-is.
- **Reads defined in this section are open on the LAN** — no token. This matches the
  deployment model (one team laptop on local Wi-Fi, not a public service) and the Q6 posture.
  The bootstrap compute endpoint (§13.7) is also open on the LAN (it is part of the
  evaluation workflow, like ingestion).

### 13.2 Latest-wins default
Every read defaults to **latest-wins**, implemented with `DISTINCT ON` over the §3.2
`result_version` rule. Per-image list endpoints accept `?history=true` to return every
stored version instead of only the latest.

### 13.3 Pagination
List endpoints are paged with `page` (0-based, default `0`) and `size` (default `50`,
max `500`). Paged bodies have shape:

```json
{ "content": [ ... ], "page": 0, "size": 50, "totalElements": 123, "totalPages": 3 }
```

### 13.4 Runs

`GET /api/v1/runs?page=0&size=50` → paged `RunSummary`:

```json
{ "content": [ { "runId": "…", "description": "…", "phase": "…", "createdAt": "2026-10-02T14:23:48Z" } ],
  "page": 0, "size": 50, "totalElements": 1, "totalPages": 1 }
```

`GET /api/v1/runs/{runId}` → `RunDetail` (server-side defaults for planning fields that were
left null at registration):

```json
{ "runId": "read-api-smoke", "description": "read api verify", "phase": null,
  "plannedTestN": 5, "plannedComparisons": 2, "bootstrapResamples": 10000, "ciLevel": 0.95,
  "createdBy": null, "createdAt": "2026-10-02T14:23:48.884678Z" }
```

### 13.5 Per-image scores

`GET /api/v1/runs/{runId}/scores?stage=batch_variant&variant=…&history=false&page=0&size=50`

- `stage` is required and must be `batch_variant` (filter with `variant=`) or `fusion`
  (filter with `strategy=`); anything else → `400 UNKNOWN_STAGE`.
- `history=true` returns every stored version; default is latest-wins per
  (image, variant-or-strategy) pair.

Paged `ImageScoreRow` rows (`variant` set for batch_variant, `strategy` for fusion,
`latencyMs` only for batch_variant):

```json
{ "content": [ { "imageId": "img001", "variant": "full_pipeline", "strategy": null,
                 "ssim": 0.86, "psnr": 26.9, "latencyMs": 31.5, "resultVersion": 1 } ],
  "page": 0, "size": 50, "totalElements": 6, "totalPages": 1 }
```

### 13.6 Ablation

`GET /api/v1/runs/{runId}/ablation` → the §5.7 seven variants, always all seven, each with
latest-wins per-image count, mean SSIM/PSNR and nearest-rank p95 latency (§5.4 rule):

```json
{ "runId": "read-api-smoke", "variants": [
    { "variant": "single_raw_frame", "n": 6, "meanSsim": 0.7316, "meanPsnr": 22.916, "p95LatencyMs": 4.6 },
    { "variant": "fusion_naive", "n": 0, "meanSsim": null, "meanPsnr": null, "p95LatencyMs": null },
    … ] }
```

### 13.7 Snapshot

`GET /api/v1/runs/{runId}/snapshot` → canonical demo-app document. Top-level keys: `run`
(§13.4 detail minus `bootstrapResamples`/`createdBy`), `ablation` (§13.6 list),
`bootstrap` (latest-wins rows from `computed_bootstrap_result`, §13.8 shape plus
`computedAt`), `learnedFusionDelta` (latest `fusion_model_diagnostic` row, or `null`),
`restormer` (latest-wins `restormer_go_nogo` means, or `null`), and `snapshotVersion` —
the lowercase hex SHA-256 of the canonical JSON serialization of the body *without* the
`snapshotVersion` field (`Canonicalizer.sha256Hex(write(body))`).

```json
{ "run": { … }, "ablation": [ … ], "bootstrap": [ … ],
  "learnedFusionDelta": null, "restormer": null,
  "snapshotVersion": "d8899496bd715a38…" }
```

### 13.8 Bootstrap compute-and-store

`POST /api/v1/runs/{runId}/bootstrap` (open on the LAN like all §13 reads — it is part of
the evaluation workflow, like ingestion).

Request (`BootstrapComputeRequest`):

```json
{ "variantA": "single_raw_frame", "variantB": "full_pipeline",
  "metrics": ["ssim", "psnr"], "seed": 42, "heldOutOnly": false }
```

- `variantA`/`variantB` required, must differ, must be one of the §5.7 seven variants.
- `metrics` optional; defaults to `["psnr", "ssim"]` when null/empty; each must be
  `psnr` or `ssim`.
- `seed` optional; when omitted the server derives one deterministically from
  `runId + comparisonKey + metric` (`comparisonKey` = `"<variantA>__vs__<variantB>"`), so the
  same request always reproduces the same result.
- `heldOutOnly` optional, default `false`; when `true` only images present in the
  `heldout_test_image` manifest are paired.

Algorithm (pure Java, `com.prelude.resultsservice.stats.PairedBootstrap`): pair the two
variants' latest-wins per-image scores by `imageId` (ordered lexicographically for
determinism), take differences `a − b`, run **10,000** resample means with the JDK-specified
`java.util.Random(seed)` sequence, and take linear-interpolation quantiles at the
Bonferroni-corrected level — family α 0.05 split across the run's `plannedComparisons`.
`MIN_PAIRED_N = 5`. Status is `significant` iff the corrected interval excludes zero,
otherwise `directional` (§9 rule).

Response (`BootstrapComputeResponse`, one entry per requested metric):

```json
{ "runId": "read-api-smoke", "comparisonKey": "single_raw_frame__vs__full_pipeline",
  "results": [ { "metric": "ssim", "pairedN": 6, "meanDelta": -0.15,
                 "ciLower": -0.16, "ciUpper": -0.14, "ciWidth": 0.02,
                 "significanceStatus": "significant", "resamples": 10000,
                 "plannedComparisons": 2, "correctedAlpha": 0.025,
                 "seed": 1379949576259236188, "resultVersion": 1 } ] }
```

Each result is stored in the `computed_bootstrap_result` table (V2 migration) with a
server-assigned monotonic `result_version` per (run, comparisonKey, metric) — the same §3.2
append-only supersede rule as scores; every §13.7 snapshot reads the latest-wins version.

Error responses (body `{code, message}`, same shape as §10 errors):

| Status | Codes |
|---|---|
| 404 | `RUN_NOT_FOUND` |
| 400 | `MISSING_VARIANTS`, `UNKNOWN_VARIANT`, `SAME_VARIANT`, `UNKNOWN_METRIC`, `UNKNOWN_STAGE`, `MALFORMED_REQUEST` |
| 409 | `NO_OVERLAPPING_IMAGES`, `INSUFFICIENT_PAIRED_N`, `INVALID_PLANNED_COMPARISONS` |

### 13.9 Ratification status

Addendum v1.1.0 drafted 2026-10-02 by Role 1's system agent; §13.1–13.3 are the only parts
the Read API implementation assumed. Ratification pending — do not build new clients against
this section until the orchestrator ratifies it.