# quantization-deploy/ — Role 5a

INT8 quantization, LiteRT on-device deployment, discard-race timeout
implementation. Package: `com.prelude.denoise`. See `docs/STATUS.md` for
build status and next action.

## Module structure

Android library module (pure **Java**) included by `capture-android` via
`include(":quantization-deploy")` in `settings.gradle.kts`.

```
src/main/java/com/prelude/denoise/
├── DenoiseModule.java          — top-level API: FusedFrame → DenoisedFrame
├── api/
│   ├── InferenceApi.java       — latency-measurement API for calibration (D6)
│   └── ThermalStateProvider.java — thermal hook interface for Role 1
├── model/
│   ├── FusedFrame.java         — input contract object (§6.3)
│   └── DenoisedFrame.java      — output contract object (§6.4)
├── tiling/
│   └── Tiler.java              — overlapping-tile inference with halo-drop + linear blend (D3)
└── timeout/
    ├── DiscardRaceRunner.java   — core discard-race mechanism (T1–T8)
    ├── InferenceRunner.java     — inference abstraction (interface, stubbed)
    ├── RaceResult.java          — race outcome with live state
    ├── RaceState.java           — CAS state machine enum
    ├── TimeSource.java          — injectable clock interface (T7)
    └── SystemTimeSource.java    — default clock (System.nanoTime)
```

### Python scripts
```
convert_smoke.py    — PyTorch → LiteRT FP32 conversion + parity gate
requirements.txt    — Python deps (ai-edge-torch, ai-edge-quantizer)
```

## Dependencies

| Dependency | Version | Why | Where pinned | Source |
|---|---|---|---|---|
| `com.google.ai.edge.litert:litert` | 2.1.5 | On-device LiteRT inference runtime | `capture-android/gradle/libs.versions.toml` | [Google Maven](https://maven.google.com/web/index.html#com.google.ai.edge.litert:litert) |
| `ai-edge-torch` (LiteRT Torch) | 0.7.2 | PyTorch to LiteRT FP32 conversion | `quantization-deploy/requirements.txt` | [PyPI](https://pypi.org/project/ai-edge-torch/) |
| `ai-edge-quantizer` | 0.4.1 | INT8 Post-Training Quantization (PTQ) | `quantization-deploy/requirements.txt` | [PyPI](https://pypi.org/project/ai-edge-quantizer/) |

> **Note:** The LiteRT dependency is declared in the version catalog but
> commented out in `build.gradle.kts` until the actual inference code is
> written (requires a `.tflite` model).

## Spec deviation: NNAPI → LiteRT accelerators

The spec says "NNAPI delegate", but **NNAPI is deprecated since Android 15**
(API 35). This module uses LiteRT 2.x with the Interpreter API (CPU/XNNPACK
baseline). GPU/NPU acceleration via CompiledModel API is a future measured
experiment with explicit CPU fallback. See rule D2.

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

## Tiling

Overlapping-tile inference with blended edges (D3):
- Tile size 256×256, halo 17 px (DnCNN receptive field radius), blend 16 px
- Separable 1D weighting (halo-drop + linear cross-fade)
- Tiled output matches full-frame within 2.38e-7 max-abs-diff (float32 precision limit)
- Peak tile working memory: 1.75 MB

## Quantization results (smoke-test — sRGB domain)

Checkpoint: `training/dncnn_best_med.pth` (Role 4 commit `89e3ffe`)
Input domain: **sRGB** — not final (pipeline uses linear FusedFrame, rule C3)

| Metric | Value |
|---|---|
| Parity gate (PyTorch vs FP32 LiteRT) | PASS — max-abs-diff 3.10e-06, PSNR 123.94 dB |
| FP32 vs INT8 mean PSNR | 34.39 dB |
| FP32 vs INT8 mean SSIM | 0.9969 |
| FP32 vs INT8 worst PSNR | 34.36 dB |
| FP32 model size | 2,250,552 bytes |
| INT8 model size | 607,792 bytes (3.7× smaller) |
| Latency | NOT RUN — needs Primary Device (rule D5) |
| Calibration data | 100 synthetic sRGB samples (final run: real SIDD train/val via leakage guard) |

## Status

- [x] Module scaffold (build.gradle.kts, AndroidManifest.xml, source dirs)
- [x] Contract objects (FusedFrame §6.3, DenoisedFrame §6.4)
- [x] Discard-race timeout mechanism (DiscardRaceRunner + 14 tests)
- [x] Single-flight policy (DenoiseModule + 4 tests)
- [x] Tiling with blended edges (Tiler + 3 tests)
- [x] Calibration API (InferenceApi for D6)
- [x] PyTorch → LiteRT FP32 Conversion & Parity Gate (`convert_smoke.py`)
- [x] INT8 quantization — smoke-test (`quantize.py`, sRGB domain)
- [ ] LiteRT inference integration (needs model bundled in assets)
- [ ] Results client (W1–W7 — separate task)
- [ ] On-device benchmarks (needs Primary Device)
- [ ] Final quantization with real SIDD data (needs linear-domain checkpoint from Role 4)
