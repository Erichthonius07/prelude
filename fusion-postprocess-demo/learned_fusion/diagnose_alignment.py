"""
diagnose_alignment.py -- is Role 2's alignment wrong, or is the data just hard?

For synthetic bursts we know the TRUE warp between every frame and the reference.
For each aligned non-reference frame this compares, in pixels at 4 image points:
  * err_applied : Role 2's warp as actually applied  (inv(homography), as in their code)
  * err_swapped : the same matrix without the inverse (detects a direction bug)
  * err_none    : doing nothing (the shift the warp is supposed to remove)
and relates them to ransacInlierRatio and to the SIDD scene.

Run from repo root:
  python fusion-postprocess-demo/learned_fusion/diagnose_alignment.py --bursts data/synthetic_bursts --limit 60
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent)]
from alignment.multi_burst_alignment import align_burst          # noqa: E402

T = np.array([[1, 0, 64], [0, 1, 64], [0, 0, 1.0]]); TI = np.linalg.inv(T)
PTS = np.array([[50, 50, 1], [450, 60, 1], [60, 450, 1], [450, 450, 1.0]]).T


def corner_err(M, M_true):
    a = M @ PTS; a = a[:2] / a[2]
    b = M_true @ PTS; b = b[:2] / b[2]
    return float(np.abs(a - b).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()
    man = json.loads((a.bursts / "manifest.json").read_text())
    rows = []
    # spread the sample over scenes instead of taking the first N (all scene 001)
    items = list(man["bursts"].items())
    step = max(1, len(items) // a.limit)
    for bid, b in items[::step][:a.limit]:
        res = align_burst(str(a.bursts / b["split"] / bid))
        if res is None:
            continue
        ref = res["referenceFrameIndex"]; Hs = [np.array(h) for h in b["homographies"]]
        for fr in res["frames"]:
            if fr["frameIndex"] == ref or fr.get("homography") is None:
                continue
            M_true = TI @ Hs[ref] @ np.linalg.inv(Hs[fr["frameIndex"]]) @ T   # frame px -> ref px
            H = np.array(fr["homography"])
            rows.append(dict(scene=b["scene"], ratio=fr["ransacInlierRatio"], aligned=fr.get("aligned") is True,
                             e_app=corner_err(np.linalg.inv(H), M_true), e_swap=corner_err(H, M_true),
                             e_none=corner_err(np.eye(3), M_true)))
    if not rows:
        raise SystemExit("no aligned frames to analyse")
    g = lambda k: np.array([r[k] for r in rows])
    app, swp, non, rat = g("e_app"), g("e_swap"), g("e_none"), g("ratio")
    print(f"frames analysed: {len(rows)}   (aligned flag true: {int(sum(r['aligned'] for r in rows))})")
    print(f"median corner error px | do-nothing {np.median(non):.2f} | Role2 applied {np.median(app):.2f} | direction-swapped {np.median(swp):.2f}")
    print(f"frames where warp < 2 px: applied {np.mean(app < 2):.0%}   swapped {np.mean(swp < 2):.0%}   do-nothing {np.mean(non < 2):.0%}")
    print(f"frames where applied warp is WORSE than doing nothing: {np.mean(app > non):.0%}")
    if len(rows) > 5:
        print(f"correlation(inlier ratio, error): {np.corrcoef(rat, app)[0, 1]:.2f}  (strongly negative = ratio is a usable confidence)")
    print("inlier-ratio threshold -> frames kept | of those, % truly aligned (<2 px)")
    for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        k = rat >= th
        print(f"  >= {th:.1f}: kept {k.mean():4.0%} | correct {np.mean(app[k] < 2):4.0%}" if k.any() else f"  >= {th:.1f}: kept   0%")
    by = defaultdict(list)
    for r in rows:
        by[r["scene"]].append(r)
    print("per scene: n | median applied err px | % under 2px | median inlier ratio")
    for sc in sorted(by):
        x = by[sc]
        print(f"  {sc}: {len(x):3d} | {np.median([r['e_app'] for r in x]):6.2f} | {np.mean([r['e_app'] < 2 for r in x]):4.0%} | {np.median([r['ratio'] for r in x]):.2f}")
    if np.median(swp) < 0.5 * np.median(app):
        print("\n>>> The un-inverted matrix matches the truth much better: Role 2's warp direction looks SWAPPED.")
    elif np.median(app) < 2 and np.mean(app < 2) > 0.8:
        print("\n>>> Role 2's warp is accurate on this data.")
    else:
        print("\n>>> Warp direction is not the problem; estimation is failing on a subset (see per-scene table).")


if __name__ == "__main__":
    main()