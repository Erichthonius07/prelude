# Prelude Android: integration guide

This version of the app has no simulation. Every number and image on screen comes from the camera, the pipeline code, or the Results Service. Where a teammate's module doesn't exist yet, a "not connected" placeholder takes the pipeline's own defined fallback path and the UI says so. It never makes up values.

## 1. What's real now

| Stage | Owner | Status in this build | Code |
|---|---|---|---|
| Burst capture (Camera2, AE/AWB/AF lock, 4–8 frames) | Role 1 | **Real** | `capture/capture/BurstCaptureManager.kt` |
| Texture-aware blur rejection (+ keep sharpest if all fail) | Role 1 | **Real** (bucket edges and thresholds come from `DeviceThresholds`) | `TextureBucketEstimator.kt`, `SharpnessEstimator.kt` |
| Alignment (ORB/FAST + RANSAC) | Role 2 | **Connection point.** The placeholder passes the reference frame only, so the decider picks `fusion_single` | `pipeline/modules/ExternalModules.kt` → `AlignmentModule` |
| `pipeline_mode` flag | Role 1 | **Real.** Decided from alignment confidence only, before fusion | `pipeline/PipelineModeDecider.kt` |
| Fusion: naive / trimmed / confidence-weighted | **Role 3 (you)** | **Real**, all three run on every multi-frame burst | `pipeline/fusion/FusionModule.kt` |
| Fusion: learned weighting | **Role 3 (you)** | **Connection point** (`FusionWeightModel`) until the model is trained | `ExternalModules.kt` |
| Denoise (TFLite INT8, NNAPI/CPU, discard-race) | Role 5a + Role 4 | **Connection point.** The placeholder passes the fused frame on | `ExternalModules.kt` → `DenoiseModule` |
| Post-process (tone map + sharpen) | **Role 3 (you)** | **Real.** Params come from the Results Service report when present | `pipeline/postprocess/PostProcessModule.kt` |
| Results Service sync + offline cache | Role 1 API, Role 3 app side | **Real client.** Endpoints are the DRAFT below | `capture/net/ResultsServiceClient.kt` |
| Live-burst metrics upload (local-first, bulk) | all stages | **Real** | `capture/net/MetricsOutbox.kt`, `pipeline/PipelineRunner.kt` |

All wiring happens in one place: **`PipelineFactory.create()`** in `pipeline/PipelineRunner.kt`.

## 2. Settings you must edit

`app/src/main/java/com/prelude/PreludeConfig.kt`
- `RESULTS_SERVICE_BASE_URL`: the team laptop's LAN IP and port, e.g. `http://192.168.0.14:8080`. The phone and laptop must be on the same Wi-Fi. Cleartext HTTP is enabled in the manifest for this.
- `DEVICE_ID`: the id the calibration tool uploaded this phone's profile under.

## 3. Stage data contracts (`pipeline/contract/StageContracts.kt`)

```
CaptureModule     → FrameBurst        frames + per-frame metadata (bitmap, varianceOfLaplacian, accepted,
                                      textureBucket, blurThreshold, timestampNs, exposureTimeNs, iso)
AlignmentModule   → AlignedFrameSet   referenceIndex (sharpest frame), frames[ sourceIndex, warped bitmap,
                                      3x3 homography, ransacInlierRatio, accepted ] + AlignmentMetrics
PipelineModeDecider (alignment confidence only) → fusion_multi | fusion_single, BEFORE fusion runs
FusionModule      → FusedFrame        bitmap, mode, strategy (null in fusion_single), framesUsed
                                      + FusionStrategyMetrics for all 4 strategies
DenoiseModule     → DenoisedFrame     bitmap, applied, timedOut, latencyMs, backend, modelId
PostProcessModule → FinalImage        bitmap, params, latencyMs
```

Rules the code enforces:
- **Fusion confidence is RANSAC's inlier ratio, never the classifier's prediction.** `AlignedFrame` has no field for the classifier output. It lives only in `AlignmentMetrics.classifierPredictions` for reporting. `FusionWeightModel.weights()` takes a parameter named `ransacInlierRatios`.
- **The denoise timeout is a discard-race inside `DenoiseModule`.** The orchestrator passes `inferenceTimeoutMs` (≤ 500 ms) and never cancels anything. It isn't governed by `pipeline_mode`.
- **Live bursts are non-scored.** They have no tripod ground truth, so the app never computes PSNR/SSIM for them. Their uploaded metrics carry `"scored": false, "source": "live_demo"`. Scored numbers on screen come only from the Results Service report and are labelled with test-set N.
- **Add-only evolution.** New fields were added to `FrameSample`, `FrameBurst` and `DeviceThresholds` with defaults. Nothing was renamed or removed, so Role 1's existing code and tests still compile.

## 4. Connecting each teammate's module

**Role 2: alignment.** Implement `AlignmentModule.align(burst, referenceIndex, thresholds)`:
1. Align only `burst.acceptedFrames` to `referenceIndex`.
2. Warp each frame's bitmap into the reference geometry, at the same size as the reference.
3. Set `ransacInlierRatio` to the real inlier ratio, and `accepted = ratio >= thresholds.alignmentConfidenceFloor`.
4. Put classifier predictions in `AlignmentMetrics.classifierPredictions`.

Then in `PipelineFactory`: `val alignment: AlignmentModule = OpenCvAlignment(...)`. This needs OpenCV for Android added to `app/build.gradle.kts` by Role 2.

**Role 5a / Role 4: denoiser.** Implement `DenoiseModule.denoise(fused, timeoutMs)`:
1. Run TFLite inference on a dedicated thread. Use the NNAPI delegate, and fall back to CPU.
2. Race it against a `timeoutMs` timer.
3. If the timer wins, return `DenoisedFrame(bitmap = fused.bitmap, applied = false, timedOut = true, …)` and discard the late result.

Put the model in `app/src/main/assets/`, add `org.tensorflow:tensorflow-lite` (+ `-gpu`/NNAPI as chosen) to `app/build.gradle.kts`, and swap it in `PipelineFactory`.

**Role 3 (you): learned fusion.** Implement `FusionWeightModel`:
- Input: aligned frame pixels (ARGB `IntArray`s, same size) plus `ransacInlierRatios`.
- Output: one non-negative weight plane per frame. `FusionModule` normalises them per pixel.

Set `learnedFusion = YourModel(...)` in `PipelineFactory`. Until then the learned strategy reports `available: false`.

## 5. Results Service API (DRAFT: Role 1 owns the final contract)

Base URL comes from `PreludeConfig`. All bodies are JSON. The app calls it only when it opens or when the sync icon (top right) is tapped. It never polls.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Reachability check |
| GET | `/api/v1/comparisons/latest?deviceId={id}` | Scored seven-variant comparison for the demo (cached on device) |
| POST | `/api/v1/metrics/batch` | Bulk upload of queued live-burst stage metrics. Header `Idempotency-Key`, and every submission has its own `submissionId` |

**Comparison report** (full example: `docs/sample_comparison_report.json`; its numbers are placeholders):
```json
{
  "schemaVersion": "0.1-draft", "generatedAt": "ISO-8601", "deviceId": "pixel-8-pro", "testSetN": 40,
  "bestFusionStrategy": "naive_mean | trimmed_mean | confidence_weighted | learned",
  "postProcessSsimContribution": 0.012,
  "learnedFusionGeneralization": { "trainSsim": 0.912, "valSsim": 0.889 },
  "postProcessParams": { "toneGamma": 0.8, "contrast": 1.08, "sharpenAmount": 0.6, "sharpenRadius": 1, "source": "grid-search …" },
  "variants": [ { "id": "single_raw | naive_mean | trimmed_mean | confidence_weighted | learned_fusion | full_pipeline | single_frame_fallback",
                  "label": "…", "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "p95LatencyMs": 0,
                  "deltaPsnrVsBaseline": 0.0, "ci95": [lo, hi], "significance": "baseline | significant | directional",
                  "perImage": [ { "imageId": "…", "psnr": 0.0, "ssim": 0.0 } ] } ],
  "stageGains": [ { "stage": "align | fuse | ai denoise | post-process", "deltaPsnr": 0.0, "ci95": [lo, hi] } ]
}
```

**Metrics batch:**
```json
{ "schemaVersion": "0.1-draft", "deviceId": "pixel-8-pro",
  "submissions": [ { "submissionId": "uuid", "schemaVersion": "0.1-draft", "source": "live_demo", "scored": false,
                     "deviceId": "…", "burstId": "burst-…", "stage": "alignment | pipeline_mode | fusion | denoise | postprocess",
                     "recordedAt": "ISO-8601", "metrics": { … } } ] }
```

How the app uses the report:
- `bestFusionStrategy` picks which fusion output goes to the denoiser.
- `postProcessParams` replaces the post-process defaults.
- `variants` and `stageGains` fill the results and ablation tabs.

If the service is unreachable, the last good report is shown from `comparison_cache.json` and the band says "offline / cached results from HH:mm". Queued metrics stay in `metrics_outbox.jsonl` until a batch upload succeeds.

**Ask Role 1:** confirm or rename these field names before Phase 2 ends. Changing them later is a small edit in `ComparisonReport.parse` and `MetricsOutbox.record`.

## 6. Testing without Role 1's server

On the laptop, from `docs/tools/`:
```
python mock_results_service.py ../sample_comparison_report.json
```
Set `RESULTS_SERVICE_BASE_URL` to `http://<laptop-ip>:8080`, run the app, and tap the sync icon.
- The results and ablation tabs fill in.
- Turn the laptop's Wi-Fi off and reopen the app: it shows the cached copy.
- Each finished burst queues metrics, and they upload on the next sync.

## 7. What you'll see on the phone today

With alignment and denoise not yet connected:
- Every burst goes `fusion_single`.
- The fuse and denoise rows show "!" with "not connected".
- The results split view shows the raw reference frame against the post-processed output.

As soon as Role 2's module is plugged in, multi-frame bursts run all three hand-crafted fusion strategies, and you can flip between them on the results tab.
