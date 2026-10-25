"""
evaluate_alignment.py  --  independent evaluation of the alignment module (Role 2).

WHY: the inlier ratio is the module's own confidence, and the classifier is trained on it
(a circularity the spec asks to be stated). Every metric derived from the ratio alone says
nothing about whether the warps are RIGHT. This script measures correctness independently,
then shows whether the ratio can be trusted, and derives a confidence floor from that.

It calls the module's real  align_burst()  and does not modify it.
Place in alignment/ (or anywhere); run from the repo root.

TWO MODES (auto-detected)
  1. REAL bursts (no ground truth):   --root <folder whose sub-folders are bursts>
       frame is "good" if the warp moved it closer to the reference
       (blurred mean-abs-difference drops by > 2%, interior only, downscaled to <=1024 px).
       This is a screening proxy: necessary, not sufficient, for an accurate warp.
  2. SYNTHETIC bursts with a manifest.json (from synth_bursts.py): TRUE motion is known
       frame is "good" if the applied warp is within --good-px (default 2) of the truth
       at 4 image points.

OUTPUT
  console summary + <out>/alignment_eval.json + <out>/alignment_eval_frames.csv (one row per frame)
  Headline: alignment SUCCESS RATE with a 95% Wilson interval, per-ratio-bin reliability,
  and a SUGGESTED FLOOR = lowest ratio threshold where >= --target-precision of kept frames
  are good (or an explicit statement that no threshold reaches it).

  python alignment/evaluate_alignment.py --root <bursts> [--out alignment_eval] [--limit 0]
"""
import argparse, csv, json, math, os, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from alignment.multi_burst_alignment import align_burst          # noqa: E402

EXT = (".png", ".jpg", ".jpeg")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def burst_dirs(root: Path):
    for d, subdirs, files in os.walk(root):
        subdirs[:] = [s for s in subdirs if not s.endswith("_clean")]   # skip synthetic target folders
        if sum(f.lower().endswith(EXT) for f in files) >= 2:
            yield Path(d)


def prep(im, scale):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if scale < 1:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (0, 0), 2.0 if scale >= 1 else 1.5)


def inner(g):
    m = max(8, int(0.05 * min(g.shape)))
    return g[m:-m, m:-m]


def true_err(applied, Mt, size):
    P = np.array([[.1, .1, 1], [.88, .12, 1], [.12, .88, 1], [.88, .88, 1.0]]).T * np.array([[size], [size], [1]])
    a = applied @ P; a = a[:2] / a[2]; b = Mt @ P; b = b[:2] / b[2]
    return float(np.abs(a - b).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("alignment_eval"))
    ap.add_argument("--limit", type=int, default=0, help="max bursts (0 = all)")
    ap.add_argument("--split", default="all", choices=["all", "train", "val"],
                    help="synthetic sets only: use the train or the val scenes (tune on train, report on val)")
    ap.add_argument("--good-px", type=float, default=2.0)
    ap.add_argument("--target-precision", type=float, default=0.90)
    a = ap.parse_args()

    man = None; mpath = a.root / "manifest.json"
    if mpath.exists():
        mj = json.loads(mpath.read_text()); man = mj["bursts"]; margin = mj["margin"]; size = mj["out"]
        T = np.array([[1, 0, margin], [0, 1, margin], [0, 0, 1.0]]); TI = np.linalg.inv(T)
    mode = "synthetic (true motion)" if man else "real (align-gain proxy)"

    dirs = sorted(burst_dirs(a.root))
    if man and a.split != "all":
        dirs = [d for d in dirs if man.get(d.name, {}).get("split") == a.split]
    if a.limit:
        dirs = dirs[::max(1, len(dirs) // a.limit)][:a.limit]
    if not dirs:
        raise SystemExit(f"no burst folders (>=2 images) under {a.root}")

    rows = []
    for d in dirs:
        res = align_burst(str(d))
        if res is None:
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(EXT))
        ref = res["referenceFrameIndex"]
        ref_img = next(f["image"] for f in res["frames"] if f["frameIndex"] == ref)
        scale = min(1.0, 1024.0 / max(ref_img.shape[:2]))
        rg = inner(prep(ref_img, scale))
        Hs = [np.array(h) for h in man[d.name]["homographies"]] if man and d.name in man else None
        for fr in res["frames"]:
            j = fr["frameIndex"]
            if j == ref:
                continue
            row = dict(burst=d.name, frame=j, ratio=float(fr.get("ransacInlierRatio") or 0.0),
                       aligned_flag=bool(fr.get("aligned")), has_H=fr.get("homography") is not None,
                       gain=None, err_px=None, good=False)
            if row["has_H"] and fr.get("image") is not None:
                raw = cv2.imread(str(d / files[j]))
                if raw is not None:
                    d_raw = np.abs(rg - inner(prep(raw, scale))).mean()
                    d_aln = np.abs(rg - inner(prep(fr["image"], scale))).mean()
                    row["gain"] = float((d_raw - d_aln) / max(d_raw, 1e-6))
                if Hs is not None:
                    Mt = TI @ Hs[ref] @ np.linalg.inv(Hs[j]) @ T
                    row["err_px"] = true_err(np.linalg.inv(np.array(fr["homography"])), Mt, size)
                    row["good"] = row["err_px"] < a.good_px
                else:
                    row["good"] = row["gain"] is not None and row["gain"] > 0.02
            rows.append(row)
    if not rows:
        raise SystemExit("no frames analysed")

    n = len(rows); k = sum(r["good"] for r in rows)
    lo, hi = wilson(k, n)
    ratio = np.array([r["ratio"] for r in rows]); good = np.array([r["good"] for r in rows])
    bursts = {}
    for r in rows:
        bursts.setdefault(r["burst"], []).append(r["good"])
    ok_bursts = int(sum(np.mean(v) >= 0.5 for v in bursts.values()))

    print(f"mode: {mode}\nbursts: {len(bursts)}   frame pairs (non-reference): {n}")
    print(f"frames with no homography: {sum(not r['has_H'] for r in rows)}   "
          f"flagged aligned=True: {sum(r['aligned_flag'] for r in rows)}")
    print(f"\nALIGNMENT SUCCESS RATE: {k / n:.1%}   (95% CI {lo:.1%} - {hi:.1%})   "
          f"[good = {'within %.1f px of truth' % a.good_px if man else 'warp moved frame closer to reference'}]")
    print(f"bursts where >= half the frames align well: {ok_bursts}/{len(bursts)}")
    print(f"mean / median inlier ratio: {ratio.mean():.1%} / {np.median(ratio):.1%}   (descriptive only; says nothing about correctness)")

    print("\nIs the inlier ratio a trustworthy confidence?  (share of frames that are GOOD, by ratio bin)")
    for blo, bhi in ((0, .3), (.3, .5), (.5, .75), (.75, 1.01)):
        m = (ratio >= blo) & (ratio < bhi)
        if m.any():
            print(f"  ratio {blo:.2f}-{min(bhi, 1):.2f}: n={m.sum():4d}  good={good[m].mean():5.0%}")
    ths = [round(x, 2) for x in np.arange(0.0, 0.96, 0.05)]
    table = []
    for th in ths:
        m = ratio >= th
        table.append(dict(threshold=th, kept=float(m.mean()), precision=float(good[m].mean()) if m.any() else None))
    floor = next((t["threshold"] for t in table if t["precision"] is not None and t["precision"] >= a.target_precision
                  and t["kept"] >= 0.05), None)
    corr = float(np.corrcoef(ratio, good.astype(float))[0, 1]) if good.std() > 0 and ratio.std() > 0 else float("nan")
    print(f"\ncorrelation(ratio, good): {corr:+.2f}")
    if floor is None:
        print(f"SUGGESTED FLOOR: none. No ratio threshold gives >= {a.target_precision:.0%} good frames while keeping >= 5%.")
        print("  => on this data the inlier ratio cannot separate good from bad frames; report that.")
    else:
        t = next(x for x in table if x["threshold"] == floor)
        print(f"SUGGESTED FLOOR: {floor:.2f}  (keeps {t['kept']:.0%} of frames, {t['precision']:.0%} of them good)")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "alignment_eval.json").write_text(json.dumps(dict(
        mode=mode, bursts=len(bursts), frames=n, success_rate=k / n, success_ci95=[lo, hi],
        bursts_half_good=ok_bursts, corr_ratio_good=corr, suggested_floor=floor,
        target_precision=a.target_precision, threshold_table=table), indent=2))
    with open(a.out / "alignment_eval_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out / 'alignment_eval.json'} and alignment_eval_frames.csv")


if __name__ == "__main__":
    main()