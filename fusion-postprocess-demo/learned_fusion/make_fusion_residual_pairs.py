"""
make_fusion_residual_pairs.py -- Role 3 -> Role 4 handover (Role 4 spec 4.2).

Runs synthetic bursts through the REAL pipeline (Role 2 align_burst -> adapter -> Role 3 fusion)
and saves, for each burst and fusion strategy:
    <out>/<split>/<strategy>/<burstId>_fused.png     fused noisy frame (what the denoiser sees in production)
    <out>/<split>/clean/<burstId>_clean.png          clean reference, in the SAME frame coordinates
    <out>/handover_manifest.json                     everything Role 4 needs to use, filter and audit the pairs

Details that matter
  * Target = clean image warped to the reference frame Role 2 chose (render(clean, H_ref)); the fused output
    lives in that frame's coordinates, so pairing against any other view would be misaligned by design.
  * Aligned frames have empty borders where the warp pulled in nothing, so every image is cropped by --inset
    pixels on each side (recorded in the manifest). Do not train on the borders.
  * Frames below --floor (RANSAC inlier ratio, contract 6.2) are excluded before fusion; the reference stays.
    The classifier's confidence is never read.
  * Split: by SIDD scene (from the manifest). Override with --val-scenes 004,009 to match Role 4's own held-out
    scenes, otherwise validation pairs may come from scenes their model trained on.
  * Not used: Role 1's held-out test set. Nothing here touches team captures; sources are SIDD clean images.

Run from repo root:
  python fusion-postprocess-demo/learned_fusion/make_fusion_residual_pairs.py --bursts data/synthetic_bursts --out data/fusion_residual --strategy all
"""
import argparse, hashlib, json, sys
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent)]
from synth_bursts import render                                        # noqa: E402
from fusion.adapter import alignedset_to_arrays                         # noqa: E402
from fusion.strategies import naivemeanfusion, trimmedmeanfusion        # noqa: E402
from alignment.multi_burst_alignment import align_burst                 # noqa: E402
try:
    from fusion.strategies import confidenceweightedfusion              # noqa: E402
except ImportError:
    confidenceweightedfusion = None

STRATS = ("naive", "trimmed", "confidence")


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--strategy", default="all", choices=("all",) + STRATS)
    ap.add_argument("--floor", type=float, default=0.0, help="min inlier ratio for a non-reference frame to be fused")
    ap.add_argument("--trimfrac", type=float, default=0.4)
    ap.add_argument("--inset", type=int, default=32)
    ap.add_argument("--val-scenes", default="", help="comma list of scene numbers (e.g. 004,009) to use as validation")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    strategies = STRATS if a.strategy == "all" else (a.strategy,)
    if "confidence" in strategies and confidenceweightedfusion is None:
        raise SystemExit("confidenceweightedfusion is not in fusion/strategies.py yet - paste it in, or use --strategy naive|trimmed")

    man = json.loads((a.bursts / "manifest.json").read_text())
    margin = man["margin"]
    val_scenes = {s.strip() for s in a.val_scenes.split(",") if s.strip()}
    items = list(man["bursts"].items())
    if a.limit:
        items = items[::max(1, len(items) // a.limit)][:a.limit]
    ins = a.inset
    crop = lambda im: im[ins:-ins, ins:-ins] if ins else im

    entries, skipped, gains = [], {"no_alignment": 0, "error": 0}, {s: [] for s in strategies}
    for i, (bid, b) in enumerate(items, 1):
        split = ("val" if b["scene"] in val_scenes else "train") if val_scenes else b["split"]
        try:
            res = align_burst(str(a.bursts / b["split"] / bid))
            if res is None:
                skipped["no_alignment"] += 1; continue
            ref = res["referenceFrameIndex"]
            frames, ratios = alignedset_to_arrays(res)                       # BGR float32 0-1, ransacInlierRatio only
            keep = ratios >= a.floor
            if not keep.any():
                skipped["no_alignment"] += 1; continue
            frames_k, ratios_k = frames[keep], ratios[keep]
            clean640 = cv2.imread(str(a.bursts / f"{b['split']}_clean" / f"{bid}.png"))
            target = render(clean640, np.array(b["homographies"][ref])).astype(np.float32) / 255.0
            target_c = crop(target)
            cdir = a.out / split / "clean"; cdir.mkdir(parents=True, exist_ok=True)
            cpath = cdir / f"{bid}_clean.png"
            cv2.imwrite(str(cpath), np.round(target_c * 255).astype(np.uint8))
            ref_pos = int(np.argmax(ratios_k))                                # reference has ratio 1.0
            psnr_ref = psnr(crop(frames_k[ref_pos]), target_c)
            for s in strategies:
                if s == "naive":
                    fused = naivemeanfusion(frames_k)
                elif s == "trimmed":
                    fused = trimmedmeanfusion(frames_k, a.trimfrac) if len(frames_k) >= 3 else naivemeanfusion(frames_k)
                else:
                    fused = confidenceweightedfusion(frames_k, ratios_k)
                fused_c = crop(fused)
                fdir = a.out / split / s; fdir.mkdir(parents=True, exist_ok=True)
                fpath = fdir / f"{bid}_fused.png"
                cv2.imwrite(str(fpath), np.round(np.clip(fused_c, 0, 1) * 255).astype(np.uint8))
                p = psnr(fused_c, target_c); gains[s].append(p - psnr_ref)
                entries.append(dict(burstId=bid, split=split, scene=b["scene"], source=b["source"], strategy=s,
                                    fused=str(fpath.relative_to(a.out)), clean=str(cpath.relative_to(a.out)),
                                    fused_sha256=sha(fpath), clean_sha256=sha(cpath),
                                    frames_in_burst=int(len(frames)), frames_fused=int(len(frames_k)),
                                    reference_index=int(ref), ratios_fused=[round(float(r), 4) for r in ratios_k],
                                    psnr_fused_db=round(p, 2), psnr_reference_frame_db=round(psnr_ref, 2)))
        except Exception as e:
            skipped["error"] += 1
            if skipped["error"] <= 3:
                print(f"  error on {bid}: {type(e).__name__}: {e}")
        if i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "handover_manifest.json").write_text(json.dumps(dict(
        description="Fusion-residual pairs for Role 4 (spec 4.2). fused = pipeline output, clean = ground truth in the same coordinates. "
                    "Colour order BGR as files (cv2). 8-bit PNG. Crop of 'inset' px removed from every side. "
                    "Sources are SIDD clean images; no Role 1 held-out images.",
        inset=a.inset, floor=a.floor, trimfrac=a.trimfrac, val_scenes_override=sorted(val_scenes), skipped=skipped,
        pairs=entries), indent=1))
    n_b = len({e["burstId"] for e in entries})
    print(f"\nbursts written: {n_b} of {len(items)}   skipped: {skipped}")
    for sp in ("train", "val"):
        print(f"  {sp}: {len({e['burstId'] for e in entries if e['split'] == sp})} bursts")
    print("\nPSNR gain of the fused frame over the single reference frame (dB, median / share of bursts improved):")
    for s in strategies:
        g = np.array(gains[s])
        if len(g):
            print(f"  {s:11s} {np.median(g):+5.2f} dB / {np.mean(g > 0):4.0%}   (n={len(g)})")
    print(f"\nwrote {a.out / 'handover_manifest.json'}")


if __name__ == "__main__":
    main()