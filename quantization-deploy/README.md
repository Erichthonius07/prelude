# quantization-deploy/ — Role 5a

INT8 quantization, LiteRT on-device deployment, discard-race timeout
implementation. Package: `com.prelude.denoise`. See `docs/STATUS.md` for
build status and next action.

## Module structure

Android library module (pure **Java**) included by `capture-android` via
`include(":quantization-deploy")` in `settings.gradle.kts`.

```
src/main/java/com/prelude/denoise/
├── DenoiseModule.java          — top-level API: FusedFrame → RaceResult (§6.3 → §6.4)
├── api/
│   ├── InferenceApi.java       — latency-measurement API for calibration (D6)
│   └── ThermalStateProvider.java — thermal hook interface for Role 1
├── model/
│   ├── FusedFrame.java         — input contract object (§6.3)
│   ├── DenoisedFrame.java      — output contract object (§6.4)
│   ├── FrameGeometry.java      — local width/height/channels (contract §6.3 gap)
│   ├── LiteRtAdapter.java      — adapter interface over the LiteRT interpreter
│   ├── LiteRtAdapterImpl.java  — InterpreterApi CPU/XNNPACK implementation
│   ├── LiteRtInferenceRunner.java — whole-frame InferenceRunner (reused scratch buffers)
│   ├── ModelSource.java / FileModelSource.java — model file loading
├── tiling/
│   ├── Tiler.java              — overlapping-tile inference with halo-drop + linear blend (D3)
│   ├── TileInferenceAdapter.java — tile Inferencer, buffers reused (0 per-tile allocations)
│   └── HwcChwConverter.java    — HWC ↔ NCHW layout conversion
└── timeout/
    ├── DiscardRaceRunner.java   — core discard-race mechanism (T1–T8)
    ├── InferenceRunner.java     — synchronous inference abstraction (T1)
    ├── RaceResult.java          — race outcome with live state
    ├── RaceState.java           — CAS state machine enum
    ├── ErrorListener.java       — error/info listener interface
    ├── TimeSource.java          — injectable clock interface (T7)
    └── SystemTimeSource.java    — default clock (System.nanoTime)

src/test/java/com/prelude/denoise/    — 43 JVM tests (8 suites, rule X3)
src/androidTest/java/com/prelude/denoise/
├── model/LiteRtAdapterImplAndroidTest.java — JNI load smoke (NOT RUN on device)
├── benchmark/TileLatencyBenchmarkAndroidTest.java — tile latency harness (MEASURED on vivo V2318, see below)
└── benchmark/TileLatencyBenchmarkActivity.java — foreground host activity for the harness
```

### Python scripts
```
convert_smoke.py    — PyTorch → LiteRT FP32 conversion + parity gate
quantize.py         — INT8 PTQ (currently a smoke test on synthetic sRGB)
submit_metrics.py   — Submits metrics/latency to Results Service
evaluate.py         — Offline FP32-vs-INT8 quality evaluation (metrics JSON for submit_metrics.py)
tiler_ref.py        — numpy reference of the Java Tiler (same tile/halo/blend/T9 semantics)
requirements.txt    — Python deps (see dependency table above)
```

## Offline quality evaluation (`evaluate.py`) — real-data run NOT RUN

```
python evaluate.py --fp32 <fp32.tflite> --int8 <int8.tflite> \
    --pairs-dir <dir with input/ and ground_truth/> \
    --input-domain srgb|linear --input-kind sidd-noisy|fused-frame \
    --checkpoint-sha256 <64 hex> --dataset <name> --out <metrics.json> \
    [--model-variant cnn-v1] [--smoke]
```

- Loads each model with `ai_edge_litert` (CPU), runs FULL images through
  `tiler_ref.py` — a numpy reference of the Java Tiler validated against the
  committed Java golden fixture (`test/golden_tiler_vector.txt`, cross-language
  test `test_tiler_ref.test_golden_replay_against_java_fixture`, ≤ 1e-5; the
  Java side is `TilerGoldenVectorTest`, fixture regeneration only via
  `-Dtiler.golden.regenerate=true`).
- Per image: skimage PSNR/SSIM vs ground truth for BOTH precisions
  (`data_range=1.0`, `channel_axis=2`, outputs clamped to [0,1], rule Q4) plus
  a separate `agreement_fp32_vs_int8` block.
- Decoding is fixed by `--input-domain` (rule C3 — no conversion inside the
  script): `srgb` 8-bit → /255 (SIDD convention); `linear` → `.npy` float32
  passthrough or true 16-bit PNG /65535; 8-bit images with `linear` are
  REFUSED (ambiguous encoding). 16-bit RGB PNG cannot be faithfully decoded by
  Pillow → `.npy` is the supported linear container (open question OQ3).
- Provenance: dataset, `input_domain`, `input_kind`, `checkpoint_sha256`,
  computed SHA-256 of both `.tflite` files, file sizes, tool versions,
  image ids, `smoke` flag. Refuses to run (before any model loads) on
  unreadable images, missing pairs, or size mismatches. Latency is NOT
  measured here (device-only, rule Q4).
- Output feeds `submit_metrics.py --metrics-json` (tested offline with mocked
  HTTP). JVM test count: 49 (TilerGoldenVectorTest added).

**Real-data run: NOT RUN** — needs a real pairs directory and the final
models. Synthetic-fixture unit tests pass (`python -m unittest`, 22 Python
tests across tiler_ref/evaluate/submit_metrics).

## Dependencies

| Dependency | Version | Why | Where pinned | Source |
|---|---|---|---|---|
| `com.google.ai.edge.litert:litert` | 2.1.5 | On-device LiteRT inference runtime | `capture-android/gradle/libs.versions.toml` | [Google Maven](https://maven.google.com/web/index.html#com.google.ai.edge.litert:litert) |
| `androidx.test.ext:junit` | 1.1.5 | JUnit extensions for Android instrumented tests | `quantization-deploy/build.gradle.kts` | Google Maven |
| `androidx.test:core` | 1.5.0 | Core Android test APIs | `quantization-deploy/build.gradle.kts` | Google Maven |
| `androidx.test:runner` | 1.5.2 | `AndroidJUnitRunner` instrumentation class — was missing from the APK (device `ClassNotFoundException`, 2026-10-03); POM pairs with monitor 1.6.1 / storage 1.4.2 already on the classpath | `quantization-deploy/build.gradle.kts` | [Google Maven](https://maven.google.com/web/index.html#androidx.test:runner) |
| `ai-edge-torch` (LiteRT Torch) | 0.7.2 | PyTorch to LiteRT FP32 conversion | `quantization-deploy/requirements.txt` | [PyPI](https://pypi.org/project/ai-edge-torch/) |
| `ai-edge-quantizer` | 0.4.1 | INT8 Post-Training Quantization (PTQ) | `quantization-deploy/requirements.txt` | [PyPI](https://pypi.org/project/ai-edge-quantizer/) |
| `scikit-image` | 0.26.0 | PSNR/SSIM metrics for offline evaluation (rule Q4: `data_range=1.0`, `channel_axis=2`) | `quantization-deploy/requirements.txt` | [PyPI](https://pypi.org/project/scikit-image/) |
| `scipy` | 1.18.1 | `scipy.io.loadmat` reads the SIDD validation `.mat` files (MATLAB v5 format) | `quantization-deploy/requirements.txt` | [PyPI](https://pypi.org/project/scipy/) |

Verified in the module venv (`pip freeze`, 2026-10-03): `scikit-image==0.26.0`,
`numpy==2.5.3`, `pillow==12.3.0`, `ai-edge-litert==2.1.5`, `requests==2.34.2`.
(Note: the local venv reports `torch==2.14.1` although `requirements.txt` pins
`torch==2.12.1`, and `litert-torch==0.9.1` vs the pinned `ai-edge-torch==0.7.2` —
pre-existing venv drift, flagged here for honesty; not touched by this change.)

> **Note:** The `.tflite` and `.pth` files are ignored in `.gitignore`. The single final model file will be an exception in `.gitignore` (decision A24).

## Spec deviation: NNAPI → LiteRT accelerators

The spec says "NNAPI delegate", but **NNAPI is deprecated since Android 15**
(API 35). This module uses LiteRT 2.x with the Interpreter API (CPU/XNNPACK
baseline). GPU/NPU acceleration via CompiledModel API is a future measured
experiment with explicit CPU fallback. See rule D2.

### Inference Engine & Benchmark Metadata

- **Runtime**: LiteRT 2.1.5 (CPU baseline)
- **Threads**: `NUM_THREADS = 4` (pinned constant in `LiteRtAdapterImpl`)
- **Delegate**: `"XNNPACK requested"` (requested via `options.setUseXNNPACK(true)`)
- **Local benchmark/latency JSON metadata format**:
  ```json
  {
    "metadata": {
      "numThreads": 4,
      "delegate": "XNNPACK requested"
    },
    "events": [
      {
        "imageId": "img1",
        "precision": "fp32",
        "latencyMs": 150.0,
        "discardRaceEvent": false,
        "timeoutEvent": false
      }
    ]
  }
  ```
  *(Note: `metadata` is stored in the local latency JSON file for local benchmarking and audit; it is not sent over the wire in the HTTP payload to `/api/v1/metrics/denoise`).*

## Language decision: Java (not Kotlin)

The Android module uses **plain Java** for all on-device code. Rationale:
- Role 5a's primary toolchain is Python (conversion, quantization, evaluation)
- Java is simpler to maintain for the team than Kotlin
- No Kotlin compiler overhead; AGP 9's built-in Kotlin is used only by Role 1's code
- All thread-safety via `java.util.concurrent.atomic` (AtomicBoolean, AtomicReference, CAS)

## Discard-race timeout

Implemented per rules T1–T8. Key design:

- **CAS state machine** (PENDING → INFERENCE_WON | TIMER_WON → DISCARDED)
- Inference thread is **never killed** (T1)
- Timer win → fusion-only frame emitted immediately; late result discarded
- Single-flight policy (A26/T6): live = BUSY fusion-only; benchmark = bounded wait
- Timeout: calibrated `DeviceThresholds.inferenceTimeoutMs` (p95 + 20%, max 500 ms)
- `pipelineMode` is **never read** by timeout logic (T5/D4)
- **The whole-image tiled path runs under the same race** (2026-10-03): the
  `denoise(FusedFrame, FrameGeometry, timeoutMs, mode)` overload wraps
  `tiler.process` as an `InferenceRunner` and goes through the shared
  single-flight + `DiscardRaceRunner` core, so the timeout applies to the full
  image (not per tile), the BUSY fallback covers it, and on a timer win the
  delivered frame is fusion-only with `latencyNs` = time-to-deliver while the
  late tiled result is discarded. This overload returns `RaceResult` (like the
  non-tiled one) so callers can `awaitLateInference()` before flushing wire
  rows (T3). Covered by `DenoiseModuleTiledTest` (7 tests) with mutation
  checks pasted failing (race wrapper removed → 4 failures; timeoutMs ignored
  → latency assertion fails; abandon check removed / flag not set → tile-count
  failures; busy flag cleared early → BUSY assertion fails).
- **T9 cooperative abandon (2026-10-03)**: after the timer wins, the late
  tiled run checks an abandon flag BEFORE each tile and stops at the next tile
  boundary (the tile in flight always completes — T1). The busy flag is
  released by the late run's end (finished, abandoned, or thrown), so
  `discardRaceEvent` becomes final quickly and the interpreter frees early. A
  null result while the race is still PENDING is treated as a bug
  (`INFERENCE_ERROR`), never as an abandon.

## Tiling

Overlapping-tile inference with blended edges (D3):
- Tile size 256×256, halo 17 px (DnCNN receptive field radius), blend 16 px
- Separable 1D weighting (halo-drop + linear cross-fade)
- Tiled output matches full-frame within 2.38e-7 max-abs-diff (float32 precision limit)
- Peak tile working memory: 1.75 MB
- **No per-tile heap allocations** (2026-10-03): `TileInferenceAdapter` reuses
  its ByteBuffers and conversion arrays across tiles (returned tile array is
  reused — the Tiler consumes it synchronously); `LiteRtInferenceRunner` reuses
  its internal scratch but still allocates the returned array fresh, because it
  escapes into the delivered `DenoisedFrame` (§6.4) and must never be
  overwritten by a later run. Proven by `TileBufferReuseTest` + an
  anti-aliasing test, with mutation checks pasted failing.

## On-device latency benchmark — MEASURED (vivo V2318, Android 16, SM7550)

`src/androidTest/.../benchmark/TileLatencyBenchmarkAndroidTest.java` runs as a
**foreground** benchmark (it launches `TileLatencyBenchmarkActivity` with
`FLAG_KEEP_SCREEN_ON`; verified `topResumedActivity` during runs). All runs:
`numThreads=4`, XNNPACK requested, nearest-rank p95, thermal status recorded,
X14 freezer validity check (no `am_app_frozen` events for the test package
during the run window — see limitations below).

**Final runs (2026-10-03, `outputs/device-runs/`):**

| Model | Per-tile median / p95 / max (ms) | N | 1024×1024 uncapped (ms) | tiles | per-tile (ms, total/25) | throughDenoiseModule timeoutEvent |
|---|---|---|---|---|---|---|
| FP32 (untrained asset, latencyOnly=true) | 1869.4 / 1881.2 / 1882.1 | 30 | 45426.5 | 25 | 1817.1 | **true** |
| INT8 (`dncnn_trained_int8.tflite`, provisional sRGB) | 414.4 / 435.5 / 442.2 | 100 | 9394.2 | 25 | 375.8 | **true** |

Both models exceed the 500 ms ceiling per tile (`timeoutEvent=true` in the
through-DenoiseModule phase). FP32 numbers are `latencyOnly=true` (untrained
asset); INT8 comes from a real Role 4 checkpoint but is **PROVISIONAL**
(sRGB domain, synthetic calibration — see provenance below). None of these are
final results. Thread-count observation: `numThreads` 1/2/4/6 showed **no
observable latency difference** and exactly one running thread in every
sample (recorded observation; no root-cause claim). Earlier sweep JSONs are in
`outputs/device-runs/` (`numthreads-*.json`, `fg-*.json`, smoke runs).

**vivo freezer note (X14):** vivo's `fast_freezer` froze the instrumented test
app ~10 s into background runs, making them invalid (wall clock includes the
freeze). The working fix was whitelisting the test package:
`adb shell dumpsys deviceidle whitelist +com.prelude.denoise.test`
(revert with `-com.prelude.denoise.test`). Every reported run passed the
validity check. Also: `svc power stayon usb` + foreground activity.

Instrumentation arguments (all optional):
- `measuredRuns` — single-tile measured runs, default **100**; `warmupRuns` —
  excluded warm-up runs (count recorded), default **5**; 100 ms spacing.
- `imageSize=<w>x<h>` — ALSO benchmark a full tiled denoise of that size:
  per repeat it records BOTH `uncapped` (tiler.process directly with the real
  adapter, no race, no timeout: total ms, tile count, per-tile native ms) AND
  `throughDenoiseModule` (BENCHMARK mode, 500 ms ceiling: `timeoutEvent` is the
  feasibility signal). `repeats` default **3** with a `cooldownMs` (default
  2000) cool-down between repeats.
- `modelPath` — absolute device path to a real checkpoint.

Thermal policy (D5): **refuses to start** at `THERMAL_STATUS_MODERATE` or
above; **aborts the phase and writes partial results** (`aborted`, partial
raws, partial p95) if MODERATE is reached mid-run (in the uncapped phase via
the T9 abandon flag between tiles); status recorded before, every 10 measured
runs, and after each phase; `thermalRose=true` if any reading exceeded the
starting status.

Nearest-rank p95 (fractional ms) over the completed runs. Model default: the
bundled asset `src/androidTest/assets/dncnn-untrained-smoke.tflite`
(**gitignored** — a fresh clone must re-run `convert_smoke.py` and copy it
there). Per the C4 amendment, every JSON carries `latencyOnly: true` for the
untrained model: latency-timing evidence only, **never quality, never a final
result**.

- **MAC estimate (paper — X15: an estimate, never a conclusion):** DnCNN
  (3→64→…→64→3, 17 conv layers of 3×3, `training/model.py`) ⇒ 556,416 MACs/pixel
  ⇒ ~36.5 GMACs per 256×256 tile. The measured numbers above supersede any
  feasibility claim derived from this figure.

## DenoiseService — entry point for Role 1 / Role 3

```java
ErrorListener listener = (msg, t) -> Log.w(TAG, msg, t);
// Bundled provisional INT8 asset (see provenance below), or FileModelSource(path):
DenoiseService service = new DenoiseService(
    new AssetModelSource(getAssets(), "dncnn_int8_srgb.tflite"), listener);

FusedFrame frame = ...;                 // §6.3 (from Role 3)
FrameGeometry geom = new FrameGeometry(width, height, 3); // §6.3 gap workaround

// Calibrated path (rule T4): timeout = min(p95*1.2, 500) from the device profile
RaceResult result = service.denoise(frame, geom, timeoutMs, DenoiseModule.RunMode.LIVE);

// Until calibrated — loud-logged 500 ms ceiling (dev/test only):
RaceResult result = service.denoise(frame, geom, DenoiseModule.RunMode.LIVE);

DenoisedFrame out = result.getFrame();          // hand to PostProcess
boolean lateDropped = result.awaitLateInference(500); // before flushing wire rows (T3)
...
service.close();                                 // releases the interpreter
```

## Model provenance (bundled INT8 asset) — PROVISIONAL

- Asset: `src/main/assets/dncnn_int8_srgb.tflite` (607,792 bytes,
  SHA-256 `b3b081aeb25e6fdac93155018876a675144d3929a917b34602d2ef69ee157a79`)
  — committed under decision A24 (explicitly authorized, only this file).
- Checkpoint: `training/dncnn_best_med.pth`
  (SHA-256 `c664708b7e079433d71b27207a8fdc508ba6e7824fd5cdca94390229f0312c41`,
  Role 4 commit `89e3ffe`, run `phase3-denoise-sidd-medium-l1`).
- Converter / quantizer: `ai-edge-torch==0.7.2` / `ai-edge-quantizer==0.7.0`
  (requirements.txt pins; the local venv reports `litert-torch==0.9.1` — drift
  noted in the dependency table).
- Input domain: **sRGB** (rule C3 mismatch with linear FusedFrame — see
  limitations).
- INT8 calibration data: **real SIDD validation blocks** — 100 blocks from
  images 0–19 of the public SIDD validation split (downloaded from the official
  SIDD page, URLs + SHA-256s in `data/sidd/validation/extraction_provenance.json`;
  evaluation used the disjoint images 20–39, 50 blocks). This replaced the
  earlier synthetic-calibration INT8 (28.052 → 28.662 dB vs GT, gap to FP32
  1.183 → 0.574 dB; VALIDATION / srgb / not held-out / not final).
- **PROVISIONAL (sRGB domain, validation-block calibration).** Not a final
  artifact; the leakage-guard server check was NOT RUN (service unreachable).

## Quality evaluation — VALIDATION / srgb / not held-out / not final

Data: public SIDD validation split, downloaded from the official SIDD page
(URLs + file SHA-256s in `data/sidd/validation/extraction_provenance.json`,
generated by `sidd_extract.py` with seed 42). Split BY IMAGE: calibration =
images 0–19 (100 blocks), evaluation = images 20–39 (**50 block pairs**,
disjoint by construction, asserted). Leakage-guard server check: NOT RUN
(Results Service unreachable).

| | mean PSNR vs GT | mean SSIM vs GT |
|---|---|---|
| noisy input baseline | 24.136 | 0.346 |
| trained FP32 (`outputs/dncnn_trained_fp32.tflite`) | 29.235 | 0.572 |
| INT8, synthetic calibration (old asset) | 28.052 | 0.499 |
| **INT8, real SIDD calibration (current asset)** | **28.662** | **0.539** |
| FP32↔INT8 agreement | synthetic 35.259 → real-cal 39.109 | 0.903 → 0.958 |

Sanity: trained FP32 beats the noisy baseline by ~5.1 dB (conversion/domain
OK). The real-calibration INT8 replaced the synthetic one (higher mean PSNR
AND smaller gap to FP32). All numbers: skimage, data_range=1.0,
channel_axis=2, outputs clamped [0,1], full 256×256 blocks (1 tile/block).
**Not final results** (sRGB domain, provisional calibration).

On-device `DenoiseService` smoke (vivo V2318, foreground): `OK (1 test)` —
one 300×280 frame through the bundled asset, structural asserts only;
`timeoutEvent=true` at the uncalibrated 500 ms ceiling (4 tiles ≈ 4×415 ms),
`latencyMs=501.5` (time-to-deliver). JSON in `outputs/device-runs/
denoise-service-device.json`.

## Known limitations (observations, no root-cause claims)

1. **FusedFrame size/layout gap** — §6.3 defines no width/height/layout;
   callers must pass `FrameGeometry` alongside (draft notices to Role 1/3).
2. **sRGB vs linear input** — bundled model is sRGB-domain; FusedFrame is
   linear per §6.3. No conversion is performed anywhere (C3). Pending Role 4's
   linear-domain checkpoint.
3. **500 ms ceiling vs measured tile time** — measured per-tile medians
   (FP32 1869.4 ms, INT8 414.4 ms) both exceed 500 ms; `timeoutEvent=true` on
   every full-image run means fusion-only delivery would be the norm at
   1024×1024. Ceiling discussion is a team-lead decision.
4. **Thread count had no observable effect** — `numThreads` 1/2/4/6 showed no
   latency difference and one running thread in every sample.
5. **vivo freezer** — background instrumented runs were frozen by
   `fast_freezer` ~10 s in (runs invalid); fixed by
   `dumpsys deviceidle whitelist +com.prelude.denoise.test` + foreground
   activity; X14 validity check after every device run.
6. **Uncalibrated timeout** — no calibrated device profile exists; the 500 ms
   ceiling default is loud-logged and must not be used for graded runs (T4).
7. **Provisional model** — synthetic calibration, sRGB domain (above).

## Quantization results (smoke-test — sRGB domain)

Checkpoint: `training/dncnn_best_med.pth` (Role 4 commit `89e3ffe`)
Input domain: **sRGB** — not final (pipeline uses linear FusedFrame, rule C3)

| Metric | Value |
|---|---|
| Parity gate (PyTorch vs FP32 LiteRT) | PASS — max-abs-diff 3.10e-06, PSNR 123.94 dB |
| FP32 vs INT8 mean PSNR | 34.39 dB (Smoke Test) |
| FP32 vs INT8 mean SSIM | 0.9969 (Smoke Test) |
| FP32 vs INT8 worst PSNR | 34.36 dB |
| FP32 model size | 2,250,552 bytes |
| INT8 model size | 607,792 bytes (3.7× smaller) |
| Latency | MEASURED on vivo V2318 (Android 16, SM7550), foreground harness — see benchmark section; latencyOnly / provisional |
| Calibration data | 100 synthetic sRGB samples (final run: real SIDD train/val via leakage guard) |

## Status

- [x] Module scaffold (build.gradle.kts, AndroidManifest.xml, source dirs)
- [x] Contract objects (FusedFrame §6.3, DenoisedFrame §6.4)
- [x] Discard-race timeout mechanism (DiscardRaceRunner + 15 tests)
- [x] Single-flight policy (DenoiseModule + 5 tests)
- [x] Tiled full-image path under the discard-race timeout, incl. T9
      cooperative abandon between tiles (DenoiseModuleTiledTest, 7 tests +
      mutation checks) — 2026-10-03
- [x] Tiling with blended edges (Tiler + 5 tests)
- [x] Per-tile buffer reuse (TileBufferReuseTest + aliasing test, mutation checks) — 2026-10-03
- [x] Calibration API (InferenceApi for D6)
- [x] PyTorch → LiteRT FP32 Conversion & Parity Gate (`convert_smoke.py`)
- [x] INT8 quantization — smoke-test (`quantize.py`, sRGB domain)
- [x] LiteRT inference integration (HWC↔NCHW, lifecycle) — unit-tested; MEASURED on device (latencyOnly)
- [x] Device latency harness (foreground activity, thermal abort, full-image mode) — MEASURED on vivo V2318: FP32 tile median 1869.4 ms / INT8 414.4 ms; 1024×1024 uncapped 45.4 s / 9.4 s; both exceed the 500 ms ceiling (2026-10-03)
- [x] Results client (`submit_metrics.py`, 8 Python tests)
- [x] Offline quality evaluation (`evaluate.py` + `tiler_ref.py`, 14 Python tests,
      Java golden fixture cross-check; real-data run NOT RUN) — 2026-10-03
- [ ] On-device benchmarks (needs Primary Device; p95 decides the 500 ms ceiling question)
- [ ] Final quantization with real SIDD data (needs linear-domain checkpoint from Role 4)

JVM test count: **49** across 9 suites — 48 verified via
`./gradlew :quantization-deploy:cleanTestDebugUnitTest :quantization-deploy:testDebugUnitTest`
(2026-10-03); the added `TilerGoldenVectorTest` (1) is verified via plain
`javac` + `JUnitCore` and awaits the next full Gradle run. Python tests:
**22** (test_tiler_ref 6 incl. the Java golden replay, test_evaluate 8,
test_submit_metrics 8) — run with `.venv/bin/python -m unittest`.

Note: images smaller than one tile are zero-padded; edge pixels may differ from a same-size run; real frames are larger than a tile.
