# Burst Low-Light Denoising & Fusion Engine

Multi-frame burst photography pipeline: Capture → Align → Fuse → Denoise (AI) →
Post-process → Final Image. See `Prelude_Burst_Denoising_Engine.pdf` and
`docs/team-map.md` for full project/team context.

## Repo layout

```
results-service/        Role 1 — Spring Boot + PostgreSQL Results Service (Dockerized)
capture-android/        Role 1 — Camera2 burst capture (Android Studio project, not Dockerized)
calibration-tool/       Role 1 — deterministic 4-threshold calibration
batch-runner/           Role 1 — headless 7-variant ablation runner
alignment/               Role 2 — RANSAC alignment + learned confidence classifier
fusion-postprocess-demo/ Role 3 — 4-way fusion, post-processing, demo app UI
training/                Role 4/5a/5b — shared Python/ML environment (Dockerized)
quantization-deploy/     Role 5a — INT8 quantization, TFLite/NNAPI deployment
restormer-stretch/       Role 5b — Restormer conversion + fine-tuning + go/no-go
docs/                     Data contract, team map
```

## Pinned versions

| Component | Version |
|---|---:|
| Java | 25 (LTS) |
| Spring Boot | 4.1.0 |
| Maven | 3.9.16 |
| PostgreSQL | 18-alpine |
| Python | 3.13 |
| PyTorch / TensorFlow | 2.12.1 / 2.20.0 (team must pick one — see below) |

Full rationale and compatibility notes (including a real issue with TF 2.20
deprecating `tf.lite` that affects Role 5a/5b) in `docs/versions.md`. Update
there first if any version changes.

## Getting started

1. **Clone the repo**, then copy the env template:
   ```bash
   cp .env.example .env
   # edit .env with real (local-only) Postgres credentials
   ```

2. **Bring up the Results Service + Postgres**:
   ```bash
   docker compose up --build
   ```
   Verify it's alive: `curl http://localhost:8080/` → `{"service":"results-service","status":"up",...}`

   This is the only always-on stack. It should run on one team-controlled
   laptop over local Wi-Fi (per spec — not cloud, not on-device), reachable
   by every teammate's Android device on the same network.

3. **Training/ML work** (Roles 2-5b) uses a separate, on-demand container —
   see `training/README.md`. Don't add it to `docker-compose.yml`; it's heavy
   and not something everyone needs running all the time.

4. **Android capture app** (`capture-android/`) is a normal Android Studio /
   Gradle project — not Dockerized, since it needs the Android SDK and a
   device/emulator.

## Branching

Single monorepo, feature branches per module (`role2/alignment`,
`role3/fusion`, etc.). Any change to `docs/data-contract.md` requires Role 1's
PR review (see `docs/team-map.md`).

## Status

- [x] Repo + monorepo structure
- [x] Docker Compose skeleton (Postgres + Results Service, boots but no real endpoints yet)
- [x] Training container skeleton
- [ ] PyTorch vs. TensorFlow decision (see `docs/versions.md` for the LiteRT consideration)
- [ ] Data contract (`docs/data-contract.md`) — next up
- [ ] Real ingestion API
- [ ] Calibration tool
- [ ] Capture module
- [ ] Batch runner
