# quantization-deploy/ — Role 5a

INT8 quantization, LiteRT on-device deployment, discard-race timeout
implementation. Package: `com.prelude.denoise`. See `docs/STATUS.md` for
build status and next action.

## Module structure

Android library module included by `capture-android` via
`include(":quantization-deploy")` in `settings.gradle.kts`.

```
src/main/kotlin/com/prelude/denoise/
├── DenoiseModule.kt          — top-level API: FusedFrame → DenoisedFrame
├── api/InferenceApi.kt       — latency-measurement API for calibration (D6)
├── model/
│   ├── FusedFrame.kt         — input contract object (§6.3)
│   └── DenoisedFrame.kt      — output contract object (§6.4)
└── timeout/
    ├── DiscardRaceRunner.kt   — core discard-race mechanism (T1–T8)
    ├── InferenceRunner.kt     — inference abstraction (stubbed)
    ├── RaceResult.kt          — race outcome with live state
    ├── RaceState.kt           — CAS state machine enum
    └── TimeSource.kt          — injectable clock
```

## Dependencies

| Dependency | Version | Why | Where pinned | Source |
|---|---|---|---|---|
| `com.google.ai.edge.litert:litert` | 2.1.5 | On-device LiteRT inference runtime | `capture-android/gradle/libs.versions.toml` | [Google Maven](https://maven.google.com/web/index.html#com.google.ai.edge.litert:litert) |
| `org.jetbrains.kotlinx:kotlinx-coroutines-android` | 1.9.0 | Coroutines for background threading | `capture-android/gradle/libs.versions.toml` | [Maven Central](https://central.sonatype.com/artifact/org.jetbrains.kotlinx/kotlinx-coroutines-android) |
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

## Discard-race timeout

Implemented per rules T1–T8. Key design:

- **CAS state machine** (PENDING → INFERENCE_WON | TIMER_WON → DISCARDED)
- Inference thread is **never killed** (T1)
- Timer win → fusion-only frame emitted immediately; late result discarded
- Single-flight policy (A26/T6): live = BUSY fusion-only; benchmark = bounded wait
- Timeout: calibrated `DeviceThresholds.inferenceTimeoutMs` (p95 + 20%, max 500 ms)
- `pipelineMode` is **never read** by timeout logic (T5/D4)

## Status

- [x] Module scaffold (build.gradle.kts, AndroidManifest.xml, source dirs)
- [x] Contract objects (FusedFrame §6.3, DenoisedFrame §6.4)
- [x] Discard-race timeout mechanism (DiscardRaceRunner + tests)
- [x] Single-flight policy (DenoiseModule + tests)
- [x] Calibration API (InferenceApi for D6)
- [x] PyTorch → LiteRT FP32 Conversion & Parity Gate (`convert_smoke.py`)
- [ ] LiteRT inference integration (blocked on W1: trained checkpoint)
- [ ] Tiling logic (D3 — needs model + memory measurements)
- [ ] Results client (W1–W7 — separate task)
- [ ] On-device benchmarks (needs Primary Device)
