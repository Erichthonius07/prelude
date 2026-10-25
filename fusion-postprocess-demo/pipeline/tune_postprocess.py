"""
tune_postprocess.py -- Role 3: pick post-processing parameters by SSIM search, then
report the stage's OWN per-image SSIM contribution (spec 3.3).

Input: Role 3's handover_manifest.json from make_fusion_residual_pairs.py (fused + clean
pairs). Post-processing really runs on the DenoisedFrame; until Role 4/5a's denoiser
output exists, tune on the fused frames and re-run once denoised frames are available.

Tuning images vs scoring images: by default the search runs on --tune-split (train) and the
contribution is reported on --report-split (val), so the reported number isn't scored on the
images it was tuned on. The spec text says "search on the fixed test set"; if the team keeps
that, pass the same split to both and say so in report section 5.4.

Run from repo root:
  python fusion-postprocess-demo/pipeline/tune_postprocess.py --manifest data/fusion_residual/handover_manifest.json \
      --strategy confidence --nrandom 80 --out results/postprocess
"""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]

from pipeline.search import gridsearchpostprocess, postprocesscontribution   # noqa: E402
from common.results import make_submission, submit                          # noqa: E402


def read01(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def load_pairs(manifest, strategy, split, limit):
    root = manifest.parent
    pairs = [p for p in json.loads(manifest.read_text())["pairs"] if p["strategy"] == strategy and p["split"] == split]
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise SystemExit(f"no '{strategy}' pairs in split '{split}' of {manifest}")
    return [p["burstId"] for p in pairs], [read01(root / p["fused"]) for p in pairs], [read01(root / p["clean"]) for p in pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--strategy", default="confidence", help="which fused frames to post-process")
    ap.add_argument("--tune-split", default="train")
    ap.add_argument("--report-split", default="val")
    ap.add_argument("--nrandom", type=int, default=0, help="0 = full grid (1728 combos); N = random search")
    ap.add_argument("--limit", type=int, default=0, help="max images per split")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dataset-id", default=None)
    ap.add_argument("--device-id", default="desktop")
    ap.add_argument("--results-url", default=None)
    a = ap.parse_args()

    _, tin, tref = load_pairs(a.manifest, a.strategy, a.tune_split, a.limit)
    print(f"searching on {len(tin)} '{a.tune_split}' images ({'grid' if a.nrandom <= 0 else f'{a.nrandom} random'})")
    best = gridsearchpostprocess(tin, tref, nrandom=a.nrandom)
    print(f"best params {best['params']}  mean SSIM on tune split {best['mean_ssim']:.4f}")

    ids, rin, rref = load_pairs(a.manifest, a.strategy, a.report_split, a.limit)
    rows = postprocesscontribution(rin, rref, best["params"], ids)
    deltas = np.array([r["ssim_delta"] for r in rows])
    same = a.tune_split == a.report_split
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"params": best["params"], "tune_split": a.tune_split, "report_split": a.report_split,
              "tuned_and_scored_on_same_images": same, "n_tune": len(tin), "n_report": len(rows),
              "mean_ssim_before": float(np.mean([r["ssim_before"] for r in rows])),
              "mean_ssim_after": float(np.mean([r["ssim"] for r in rows])),
              "mean_ssim_contribution": float(deltas.mean()), "share_images_improved": float((deltas > 0).mean()),
              "top10_candidates": best["table"][:10]}
    (a.out / "postprocess_params.json").write_text(json.dumps(report, indent=2))
    dataset_id = a.dataset_id or f"dev:{a.manifest.parent.name}/{a.report_split}/{a.strategy}"
    sub = make_submission("PostProcessModule", "postprocess", rows, dataset_id, a.device_id,
                          extra={"params": best["params"], "tuned_on": a.tune_split, "input": f"fused:{a.strategy}"})
    submit(sub, a.results_url, a.out / "results_local")
    print(f"post-processing SSIM contribution on '{a.report_split}': {deltas.mean():+.4f} "
          f"(improved {np.mean(deltas > 0):.0%} of {len(rows)} images){'  [tuned on same images]' if same else ''}")
    if np.allclose(deltas, 0):
        print("search picked (near) identity: post-processing isn't helping SSIM on this data")
    print(f"wrote {a.out / 'postprocess_params.json'}")


if __name__ == "__main__":
    main()
