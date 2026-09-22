# batch-runner

Headless 7-variant ablation runner (Role 1). Runs every evaluation variant from
`docs/data-contract.md` §5.7 over the test set and submits per-image results to the
Results Service `batch_variant` stage.

## Architecture decision (CLOSED)

**batch-runner is an Android instrumented test harness / in-app headless runner —
NOT a desktop script.**

Reason: it must log **per-device thermal state** during benchmark runs via Android's
Thermal API (`PowerManager` thermal status, API 30+), which only exists on-device. This
is the same reasoning that put the calibration tool inside `capture-android`: a
capability that physically requires the device lives on the device.

`minSdk 31` (Android 12) comfortably covers the Thermal API.

## Status: SKELETON ONLY — not built

The runner can't meaningfully execute anything until the full pipeline exists:
- Alignment (Role 2)
- Fusion, 4 strategies (Role 3)
- Denoise model (Role 4) + deployment (Role 5a)

Building it now would be speculative work against modules that don't exist. It will be
implemented once those land.

## Intended shape (for when it's unblocked)

- Instrumented harness driving the real pipeline over the locked held-out set.
- For each of the 7 variants, per image: run variant, capture SSIM/PSNR/latency,
  record thermal state before/during/after.
- Submit per-image rows to `POST /api/v1/metrics/batch-variant` with a registered `runId`.
- Guard: refuses to run unless the held-out manifest is populated and the target device
  is calibrated (blur/texture threshold present).