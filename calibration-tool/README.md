# calibration-tool/ — Role 1

Single deterministic implementation of all four per-device thresholds. Can be
an in-app debug mode (see `capture-android/`) or a headless script — decide
and document once.

## Computes (per spec 2.3 / Prelude 4.2)
1. Blur threshold — texture-bucketed (low/medium/high via gradient-magnitude histogram)
2. Alignment confidence floor — from RANSAC inlier-ratio distribution
3. Minimum usable frame count — marginal SSIM gain < 0.005, or fused < best single frame
4. Inference timeout — p95 latency + 20%, hard ceiling 500ms

Auto-uploads the resulting device profile to the Results Service.
One calibrated device = "Primary Device" for the official ablation.

_Not yet implemented._
