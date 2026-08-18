# Status

Single source of truth for what's built vs. not. Every module README links
here instead of stating its own status inline, so this never goes stale in
two places at once.

| Module | Owner | Status | Blocked by | Next action |
|---|---|---|---|---|
| Repo scaffold (structure, Docker, versions) | Role 1 | **Done** | — | — |
| ML framework decision | Team | **Done** (TensorFlow 2.20.0) | — | — |
| `results-service/` — boots, connects to Postgres | Role 1 | **Done** | — | — |
| `results-service/` — real ingestion API | Role 1 | **Not started** | Data contract | Design data contract (`docs/data-contract.md`) |
| `docs/data-contract.md` | Role 1 | **Not started** | Nothing | Design it — this is the next concrete task |
| `calibration-tool/` | Role 1 | **Not started** | Nothing (can start now) | Implement the 4 threshold calculations |
| `capture-android/` | Role 1 | **Not started** | Nothing (can start now) | Scaffold Android Studio/Gradle project |
| `batch-runner/` | Role 1 | **Not started** | Data contract + a working pipeline | Wait for both |
| `alignment/` (Role 2) | Role 2 | **Not started** | Nothing (can start now) | See `Role2_Full_Spec.pdf` |
| `fusion-postprocess-demo/` (Role 3) | Role 3 | **Not started** | Role 2's alignment output | See `Role3_Full_Spec.pdf` |
| `training/` — primary CNN (Role 4) | Role 4 | **Not started** | Nothing for SIDD-only training; fusion pipeline for the residual supplement | See `Role4_Full_Spec.pdf` |
| `quantization-deploy/` (Role 5a) | Role 5a | **Not started** | Role 4's trained model | See `Role5a_Full_Spec.pdf` |
| `restormer-stretch/` (Role 5b) | Role 5b | **Not started** | Nothing for the Phase 1-2 conversion smoke-test | Run the smoke-test first — see `docs/versions.md` "Known risks" for the PyTorch-checkpoint issue to resolve before this |

## Known risks currently unresolved

See `docs/versions.md` → "Known risks" section for the one open item:
**Restormer has no official TensorFlow checkpoint**, and the team is locked
to TensorFlow. Needs a decision before Role 5b starts Phase 1-2 work.

There are no other open decisions. Every version in `docs/versions.md` is
pinned and verified; the ML framework choice is made; nothing else in this
repo is ambiguous — it's either built, or explicitly not-started with a
listed blocker and next action above.
