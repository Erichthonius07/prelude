# capture-android/ — Role 1

Android (Kotlin, Camera2 API) capture module + the demo app shell (UI itself owned by Role 3).

**Not Dockerized** — needs the Android SDK, an emulator or physical device.
Open this folder as an Android Studio project once scaffolded.

## Owns (per spec 2.1, 2.3, 2.5)
- Burst capture: 4-8 frames, locked AE/AWB/focus
- Texture-aware blur rejection (variance-of-Laplacian, texture-bucketed)
- Emergency fallback: keep sharpest frame if all frames fail threshold
- Per-frame metadata logging (ISO, exposure, timestamp, sharpness)
- Calibration tool (in-app debug mode) — see `calibration-tool/`
- pipeline_mode flag plumbing (alignment→fusion only)

_Scaffolding (Gradle project, manifest, Camera2 boilerplate) not yet generated._
