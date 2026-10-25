"""
ablation.py -- Role 3: the 4-way fusion ablation (proposal section 6, variants 2-5).

Runs naive / trimmed / confidence-weighted / learned fusion on every burst of a
split and records PER-IMAGE PSNR and SSIM against the clean target (same metric
code for every strategy). The reference (sharpest) frame alone is scored too, as a
sanity line -- the official single-raw-frame baseline is Role 1's.

Input: .npz bursts (frames, ratios, target) as written by learned_fusion/build_dataset.py.
       For the OFFICIAL numbers this must be Role 1's held-out test set converted to the
       same format, with --dataset-id set to Role 1's test-set id. Until then use the
       synthetic val split -- those numbers are development numbers only.

Writes <out>/ablation_per_image.csv, <out>/ablation_summary.json and one Results Service
submission per strategy (local JSON under <out>/results_local, uploaded if --results-url).

Run from repo root:
  python fusion-postprocess-demo/pipeline/ablation.py --data data/fusion_dataset --split val \
      --checkpoint checkpoints/learned_fusion/fusion_weightnet.weights.h5 --out results/fusion_ablation
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]

from fusion.module import fuse_arrays                               # noqa: E402
from common.metrics import computepsnr, computessim                 # noqa: E402
from common.results import make_submission, submit                  # noqa: E402


def load_npz(path, inset):
    d = np.load(path)
    frames, ratios, target = d["frames"], d["ratios"].astype(np.float32), d["target"]
    if inset:
        frames, target = frames[:, inset:-inset, inset:-inset], target[inset:-inset, inset:-inset]
    return frames.astype(np.float32) / 255.0, ratios, target.astype(np.float32) / 255.0


def paired_bootstrap(a, b, n=10000, seed=0):
    """Quick local check of mean(a-b) with a 95% CI. Role 1's service does the official, corrected version."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "n": int(len(d)),
            "verdict": "directional" if lo <= 0 <= hi else "separated (uncorrected)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, help="learned fusion checkpoint; strategy 4 is skipped without it")
    ap.add_argument("--trimfrac", type=float, default=0.4)
    ap.add_argument("--floor", type=float, default=0.0, help="RANSAC inlier-ratio floor (Role 1's calibrated value)")
    ap.add_argument("--inset", type=int, default=48)
    ap.add_argument("--dataset-id", default=None, help="defaults to dev:<data>/<split>")
    ap.add_argument("--device-id", default="desktop")
    ap.add_argument("--results-url", default=None, help="Results Service ingestion URL; local JSON only if omitted")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted((a.data / a.split).glob("*.npz"))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"no .npz bursts in {a.data / a.split}")
    model = None
    strategies = ["naive", "trimmed", "confidence"]
    if a.checkpoint:
        from learned_fusion.infer import load_model
        model = load_model(a.checkpoint); strategies.append("learned")
    else:
        print("no --checkpoint: skipping strategy 4 (learned)")
    dataset_id = a.dataset_id or f"dev:{a.data.name}/{a.split}"
    if dataset_id.startswith("dev:"):
        print("NOTE: development numbers (not Role 1's held-out test set)")

    rows = {s: [] for s in ["reference_frame"] + strategies}
    lat = {s: [] for s in strategies}
    for i, f in enumerate(files, 1):
        frames, ratios, target = load_npz(f, a.inset)
        ref = frames[int(np.argmax(ratios))]
        rows["reference_frame"].append({"imageId": f.stem, "psnr": computepsnr(ref, target), "ssim": computessim(ref, target)})
        for s in strategies:
            t0 = time.perf_counter()
            fused, info = fuse_arrays(frames, ratios, s, trimfrac=a.trimfrac, floor=a.floor, model=model)
            lat[s].append((time.perf_counter() - t0) * 1000)
            rows[s].append({"imageId": f.stem, "psnr": computepsnr(fused, target), "ssim": computessim(fused, target),
                            "framesFused": info["frames_fused"], "strategyUsed": info["strategy_used"]})
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "ablation_per_image.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["imageId"] + [f"{s}_{m}" for s in rows for m in ("psnr", "ssim")])
        for k in range(len(files)):
            w.writerow([files[k].stem] + [f"{rows[s][k][m]:.5f}" for s in rows for m in ("psnr", "ssim")])

    summary = {"datasetId": dataset_id, "n": len(files), "inset": a.inset, "trimfrac": a.trimfrac, "floor": a.floor,
               "strategies": {s: {"mean_psnr": float(np.mean([r["psnr"] for r in rows[s]])),
                                  "mean_ssim": float(np.mean([r["ssim"] for r in rows[s]])),
                                  **({"median_ms_desktop": float(np.median(lat[s]))} if s in lat else {})} for s in rows}}
    best = max(strategies, key=lambda s: summary["strategies"][s]["mean_ssim"])
    summary["best_by_mean_ssim"] = best
    summary["best_vs_others_ssim_bootstrap"] = {
        s: paired_bootstrap([r["ssim"] for r in rows[best]], [r["ssim"] for r in rows[s]]) for s in strategies if s != best}
    (a.out / "ablation_summary.json").write_text(json.dumps(summary, indent=2))

    for s in strategies:
        sub = make_submission("FusionModule", f"fusion_{s}", rows[s], dataset_id, a.device_id,
                              extra={"trimfrac": a.trimfrac, "floor": a.floor, "inset": a.inset})
        submit(sub, a.results_url, a.out / "results_local")

    print(f"\n{'strategy':16s} {'PSNR':>7s} {'SSIM':>7s}   (N={len(files)})")
    for s, v in summary["strategies"].items():
        print(f"{s:16s} {v['mean_psnr']:7.2f} {v['mean_ssim']:7.4f}")
    print(f"\nbest by mean SSIM: {best}")
    for s, b in summary["best_vs_others_ssim_bootstrap"].items():
        print(f"  {best} - {s}: {b['mean_diff']:+.4f}  CI95 [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  {b['verdict']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
