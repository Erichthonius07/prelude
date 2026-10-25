# Prelude native Android frontend

Native Kotlin/Gradle conversion of the supplied Prelude low-light burst photography UI. The project is organized to match the requested Android Studio tree and uses `com.prelude` packages throughout.

## Package map

- `com.prelude.capture.calibration`: device thresholds, calibration repository, blur/threshold calculators, and test stubs.
- `com.prelude.capture.capture`: Camera2 lifecycle and sharpness scoring.
- `com.prelude.capture.model`: burst/frame domain models.
- `com.prelude.capture.net`: offline-first results service client.
- `com.prelude.capture.ui`: native activities and the custom instrument UI.
- `com.prelude.pipeline`: pipeline mode selection and fallback reasons.

## Build

Open this folder in Android Studio and run `./gradlew assembleDebug` on a machine with the Android SDK. This sandbox has Java but no Android SDK or Gradle executable, so the project was source-checked but not compiled here.

The view layer intentionally uses Android `View`, `Canvas`, `TextureView`, and `Camera2` rather than adding a second UI framework. Add Geist and Geist Mono TTF files under `app/src/main/res/font/` and update `CaptureModule` typography loading when the font assets are available; the current code keeps a safe platform fallback.

## UI layout notes

- The app draws edge-to-edge on every API level. `MainActivity` does not pad the root for system bars; it passes the safe-area insets (system bars + display cutout) to `CaptureModule`, which paints the bar backgrounds itself and lays everything out inside the safe area.
- `CaptureModule` computes all layout rectangles in dp once per size/inset change (`computeLayout`). Drawing and touch hit-testing use the same rectangles.
- On the capture page the overlay is opaque except for the viewfinder window, and `MainActivity` maps the camera preview into that window with an aspect-correct centre-crop via `TextureView.setTransform`. The TextureView is never resized after the camera opens.

## Pipeline wiring

The shutter runs the real pipeline: Camera2 burst (AE/AWB/AF locked, texture-aware blur rejection) → alignment → `pipeline_mode` → four-way fusion → denoise → post-process. Modules owned by other roles plug in through `PipelineFactory`; until they do, their "not connected" placeholders take the pipeline's defined fallback paths. See `docs/INTEGRATION.md` for every connection point and the draft Results Service API.
