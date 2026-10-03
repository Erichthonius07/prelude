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
└── benchmark/TileLatencyBenchmarkAndroidTest.java — tile latency harness (NOT RUN on device)
```

### Python scripts
```
convert_smoke.py    — PyTorch → LiteRT FP32 conversion + parity gate
quantize.py         — INT8 PTQ (currently a smoke test on synthetic sRGB)
submit_metrics.py   — Submits metrics/latency to Results Service
requirements.txt    — Python deps (ai-edge-torch, ai-edge-quantizer)
```

## Dependencies

| Dependency | Version | Why | Where pinned | Source |
|---|---|---|---|---|
| `com.google.ai.edge.litert:litert` | 2.1.5 | On-device LiteRT inference runtime | `capture-android/gradle/libs.versions.toml` | [Google Maven](https://maven.google.com/web/index.html#com.google.ai.edge.litert:litert) |
| `androidx.test.ext:junit` | 1.1.5 | JUnit extensions for Android instrumented tests | `quantization-deploy/build.gradle.kts` | Google Maven |
| `androidx.test:core` | 1.5.0 | Core Android test APIs | `quantization-deploy/build.gradle.kts` | Google Maven |
| `ai-edge-torch` (LiteRT Torch) | 0.7.2 | PyTorch to LiteRT FP32 conversion | `quantization-deploy/requirements.txt` | [PyPI](https://pypi.org/project/ai-edge-torch/) |
| `ai-edge-quantizer` | 0.4.1 | INT8 Post-Training Quantization (PTQ) | `quantization-deploy/requirements.txt` | [PyPI](https://pypi.org/project/ai-edge-quantizer/) |

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

## On-device latency benchmark — NOT RUN (rule H1/D5)

`src/androidTest/.../benchmark/TileLatencyBenchmarkAndroidTest.java` measures
inference latency through `LiteRtAdapterImpl` (NUM_THREADS=4, XNNPACK
requested). It **compiles** (`./gradlew :quantization-deploy:assembleDebugAndroidTest`)
but has **NOT been run on any device**. Run:
`cd capture-android && ./gradlew :quantization-deploy:connectedDebugAndroidTest`
then `adb pull` the JSON path printed in the log.

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

- **MAC estimate (paper, to be replaced by the measurement):** DnCNN
  (3→64→…→64→3, 17 conv layers of 3×3, `training/model.py`) ⇒ 556,416 MACs/pixel
  ⇒ ~36.5 GMACs per 256×256 tile (36,465,266,688). At a typical 50–100 GMAC/s
  phone-CPU rate that is ~0.4–0.7 s per tile — the 500 ms ceiling (T4) would be
  exceeded on multi-tile images. **Estimate only; the device p95 decides.**

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
| Latency | NOT RUN — needs Primary Device (rule D5) |
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
- [x] LiteRT inference integration (HWC↔NCHW, lifecycle) — unit-tested; NOT RUN on a device
- [x] Single-tile device latency harness + full-image mode with thermal
      abort/args (androidTest compiles) — **NOT RUN on a device**
- [x] Results client (`submit_metrics.py`, 8 Python tests)
- [ ] On-device benchmarks (needs Primary Device; p95 decides the 500 ms ceiling question)
- [ ] Final quantization with real SIDD data (needs linear-domain checkpoint from Role 4)

Current JVM test count: **48** across 8 suites
(DenoiseModuleTest 5, DenoiseModuleTiledTest 7, InferenceApiTest 2,
LiteRtInferenceRunnerTest 8, HwcChwConverterTest 2, TileBufferReuseTest 2,
TilerTest 5, DiscardRaceRunnerTest 17) — run with
`cd capture-android && ./gradlew :quantization-deploy:cleanTestDebugUnitTest :quantization-deploy:testDebugUnitTest`.

Note: images smaller than one tile are zero-padded; edge pixels may differ from a same-size run; real frames are larger than a tile.
