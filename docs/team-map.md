# Team Ownership

| Role | Folder(s) | Owns |
|---|---|---|
| 1 | `capture-android/`, `results-service/`, `calibration-tool/`, `batch-runner/` | Capture, integration contracts, Results Service, calibration tool, orchestration flag, batch runner |
| 2 | `alignment/` | Alignment + learned alignment-confidence classifier |
| 3 | `fusion-postprocess-demo/` | Fusion (4-way) + post-processing + learned fusion model + demo app UI |
| 4 | `training/` (shared) | Primary CNN training + fusion-residual training supplement |
| 5a | `quantization-deploy/` | Quantization + deployment + discard-race timeout implementation |
| 5b | `restormer-stretch/` | Restormer stretch goal — conversion smoke-test, fine-tuning, anchored go/no-go |

## Operating procedures (Role 1 = you)
- **Data-contract disputes**: you decide, 24-hour appeal window to the full team.
- **Algorithmic/design disagreements**: team majority vote; you break a 3-3 tie.
- **Contract changes**: require your PR review before merge.
- **Weekly full-pipeline integration test**: you run it on the Primary Device, triage failures to the responsible owner.
- **If you're unavailable**: Role 3 takes over manual result aggregation.
- **If a core role (2/3/4/5a) is affected**: Results Service falls back to local-JSON logging with manual aggregation.
- **Under time pressure**: Role 5b (stretch) is deprioritized first; then the two new ML add-ons (Role 2's classifier, Role 3's learned fusion model) revert to their hand-crafted-only fallback.
