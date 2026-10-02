# Calibration Tool — Status

**Closed decision:** this is implemented as an in-app debug mode inside
`capture-android`, not a separate module or script. On-device Thermal
and TFLite API access requires running on-device, which ruled out a
desktop script.

Real implementation lives at:
`capture-android/app/src/main/java/com/prelude/capture/calibration/`

Computes four per-device thresholds:
- Blur/texture threshold (bucketed) — built
- Alignment confidence floor — stubbed, blocked on Role 2
- Minimum usable frame count — stubbed, blocked on Role 2 + Role 3
- Inference timeout (p95 + 20%, cap 500ms) — stubbed, blocked on Role 4 + Role 5a
