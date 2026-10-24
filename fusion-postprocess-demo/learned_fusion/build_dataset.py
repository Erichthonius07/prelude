"""
build_dataset.py -- Role 3: turn synthetic bursts into learned-fusion training data.

For each burst folder written by synth_bursts.py:
  1. run Role 2's REAL align_burst()           (no classifier needed for fusion)
  2. read it ONLY through fusion.adapter        (ransacInlierRatio, aligned frames)
  3. target = render(clean, H_ref): the clean image as seen from the reference
     frame Role 2 chose (fused output lives in the reference frame's coordinates)
  4. save frames (uint8), ratios, target -> <out>/<split>/<burstId>.npz

Two sanity checks guard against silent mistakes:
  * ref_check : the reference frame vs target must match the PSNR recorded at
                generation time. If not, frameIndex != file order or the
                homography bookkeeping is off -> burst is skipped and counted.
  * align_gain: for each aligned non-reference frame, PSNR(aligned, target) minus
                PSNR(raw, target) on the interior. Positive = Role 2's warp helps.
                A negative mean means the warp direction/quality is wrong.

Run from repo root:
  python fusion-postprocess-demo/learned_fusion/build_dataset.py ^
      --bursts data/synthetic_bursts --out data/fusion_dataset
"""
import argparse, json, os, sys, traceback
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent)]  # learned_fusion, fusion-postprocess-demo, repo root

from synth_bursts import render, psnr, OUT                      # noqa: E402
from fusion.adapter import alignedset_to_arrays                 # noqa: E402
from alignment.multi_burst_alignment import align_burst         # noqa: E402

INSET = 48  # ignore warp borders when comparing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    manifest = json.loads((a.bursts / "manifest.json").read_text())
    items = list(manifest["bursts"].items())
    if a.limit:
        items = items[:a.limit]
    rep = {"built": 0, "no_alignment": 0, "ref_mismatch": 0, "errors": 0,
           "frames_in": 0, "frames_kept": 0, "gains": [], "bad_align_bursts": []}
    c = slice(INSET, OUT - INSET)

    for i, (bid, b) in enumerate(items, 1):
        split = b["split"]
        bdir = a.bursts / split / bid
        try:
            res = align_burst(str(bdir))
            if res is None:
                rep["no_alignment"] += 1; continue
            ref = res["referenceFrameIndex"]
            clean = cv2.imread(str(a.bursts / f"{split}_clean" / f"{bid}.png"))
            target = render(clean, np.array(b["homographies"][ref]))

            ref_img = next(fr["image"] for fr in res["frames"] if fr["frameIndex"] == ref)
            if abs(psnr(ref_img, target) - b["frame_psnr"][ref]) > 0.5:
                rep["ref_mismatch"] += 1; continue

            files = sorted(f for f in os.listdir(bdir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
            gains = []
            for fr in res["frames"]:
                if fr["frameIndex"] == ref or fr.get("aligned") is not True:
                    continue
                raw = cv2.imread(str(bdir / files[fr["frameIndex"]]))
                gains.append(psnr(fr["image"][c, c], target[c, c]) - psnr(raw[c, c], target[c, c]))
            if gains:
                rep["gains"].extend(gains)
                if np.mean(gains) < 0:
                    rep["bad_align_bursts"].append(bid)

            frames, ratios = alignedset_to_arrays(res)          # float32 0-1, BGR
            rep["frames_in"] += len(res["frames"]); rep["frames_kept"] += len(frames)
            (a.out / split).mkdir(parents=True, exist_ok=True)
            np.savez(a.out / split / f"{bid}.npz",
                     frames=np.round(frames * 255).astype(np.uint8), ratios=ratios,
                     ref=np.int32(ref), target=target)
            rep["built"] += 1
        except Exception:
            rep["errors"] += 1
            if rep["errors"] <= 3:
                traceback.print_exc()
        if i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)}  built={rep['built']}")

    g = rep.pop("gains")
    rep["mean_align_gain_db"] = round(float(np.mean(g)), 2) if g else None
    rep["frac_frames_with_positive_gain"] = round(float(np.mean(np.array(g) > 0)), 3) if g else None
    rep["bad_align_bursts"] = rep["bad_align_bursts"][:10]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "build_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()