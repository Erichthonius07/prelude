# alignment/ - Role 2

RANSAC-based pairwise homography alignment and confidence scoring for burst frames.

## Contract & Gating
- **Transform:** x_ref ~ H * x_moving (H[2, 2] = 1.0). Direction verified on translation (Step A); rotation/affine verification pending.
- **Metric:** Pure `ransacInlierRatio` (inliers / total_matches) per contract §5.2/§6.2.
- **Unified Discard Rule:** Role 3 discards if `status != "ok"` or `ransacInlierRatio < 0.30` (provisional cutoff). Band `[0.30, 0.50)` passes with ratio attached for weighting. Role 3 must not merge a frame solely because `status == "ok"`.
- **Status Enum (Proposed for §6.2):** `ok`, `no_keypoints` (0 descriptors), `lt4_matches` (<4 mutual matches), `ransac_failed` (solver found no H).
- **Reference Failure:** Descriptors failing on reference returns `None` / exit 0 in current code. Proposed §6.2 contract fallback: burst status `no_keypoints`, `referenceFrameIndex = -1`, identity matrices with ratio 0.0.

## Benchmark Results (Synthetic Bursts, In-Sample)
Evaluated across 60 synthetic bursts (299 frame pairs) using `alignment/evaluate_ransac_alignment.py` vs git HEAD baseline:

| Split | Pairs | Burst Crashes (HEAD -> Now) | Non-OK Failures (HEAD -> Now) | < 1.0 px (HEAD vs Now) | Median Err (HEAD vs Now) | Correlation (HEAD -> Now) |
|---|---|---|---|---|---|---|
| **Train** | 219 | 6/45 (13%) -> **0/45 (0%)** | 45/219 (20.5%) -> **0/219 (0%)** | 51.6% vs **52.1%** | 0.75 px vs 0.90 px | +0.33 -> **+0.73** |
| **Val** | 80 | 0/15 (0%) -> **0/15 (0%)** | 22/80 (27.5%) -> **0/80 (0%)** | 72.5% vs **72.5%** | 0.50 px vs 0.50 px | +0.37 -> **+0.69** |
| **Combined** | 299 | 6/60 (10%) -> **0/60 (0%)** | 67/299 (22.4%) -> **0/299 (0%)** | 57.2%* vs **57.5%** | 0.79 px | **+0.72** |

*Combined baseline 57.2% is derived from the weighted average of train (51.6%) and val (72.5%). Parameters tuned observing all 60 bursts (in-sample). Milestone remains OPEN pending component ablation, held-out split, bootstrap CIs, real-burst parallax, and ARM64 timing.*
