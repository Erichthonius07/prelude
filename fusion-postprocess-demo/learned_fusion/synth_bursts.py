"""
synth_bursts.py -- Role 3: synthetic handheld bursts from SIDD clean (GT) images.

Why: SIDD Medium has only 2 noisy shots per scene, but fusion needs 4-8 frame
bursts. So we take the CLEAN image, simulate hand shake (small random
homographies) and add Poisson-Gaussian noise per frame, then write each burst as
a folder of PNGs that Role 2's align_burst() loads exactly like a real burst.

Layout written (matches align_burst: it loads EVERY .png/.jpg in the folder,
so nothing but frames may live in a burst folder):
  <out>/<split>/<burstId>/frame_00.png ... frame_NN.png     (BGR via cv2)
  <out>/<split>_clean/<burstId>.png      clean 640px source crop (BGR)
  <out>/manifest.json                    per-burst params + TRUE homographies

IMPORTANT for the training-target step: the fused output lives in the
REFERENCE frame's coordinates, and align_burst picks the reference by
sharpness. So the target for a burst is  render(clean, H_ref)  where
H_ref = manifest[burst]['homographies'][referenceFrameIndex]. Use render().

Split is by SCENE (2nd field of the SIDD folder name) -> no scene leakage.

Usage (from repo root):
  python fusion-postprocess-demo/learned_fusion/synth_bursts.py ^
      --sidd-root data/sidd/SIDD_Medium_Srgb --out data/synthetic_bursts
"""
import argparse, json, math, random
from pathlib import Path
import cv2
import numpy as np

SRC, OUT, MARGIN = 640, 512, 64          # source crop, output frame, border margin


def scene_of(folder: str) -> str:
    return folder.split("_")[1]          # '0001_001_S6_...' -> '001'


def random_homography(rng, center):
    """Small hand-shake: shift, rotation, scale, slight perspective."""
    tx, ty = rng.uniform(-6, 6, 2)
    ang = math.radians(rng.uniform(-1.0, 1.0))
    sc = rng.uniform(0.995, 1.005)
    cx, cy = center
    c, s = math.cos(ang) * sc, math.sin(ang) * sc
    A = np.array([[c, -s, cx - c * cx + s * cy + tx],
                  [s,  c, cy - s * cx - c * cy + ty],
                  [0,  0, 1]], dtype=np.float64)
    A[2, 0], A[2, 1] = rng.uniform(-2e-5, 2e-5, 2)
    return A


def render(clean_src, H):
    """Warp the 640px clean crop with H and centre-crop to 512 (no noise)."""
    w = cv2.warpPerspective(clean_src, H, (SRC, SRC), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    return w[MARGIN:MARGIN + OUT, MARGIN:MARGIN + OUT]


def add_noise(img_u8, rng, K, sigma):
    """Poisson (shot) + Gaussian (read) noise on 0-1 values; K = photons/unit."""
    x = img_u8.astype(np.float32) / 255.0
    y = rng.poisson(np.clip(x, 0, 1) * K).astype(np.float32) / K
    y += rng.normal(0.0, sigma, x.shape).astype(np.float32)
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)


def psnr(a, b):
    m = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * math.log10(255.0 ** 2 / max(m, 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidd-root", required=True, type=Path, help="folder containing Data/")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bursts-per-image", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of SCENES held out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N source images")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth clean image (spreads a small run over all scenes)")
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help="scales noise std: 1.0 = default (~20 dB frames), 0.3 = much cleaner frames")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    pyr = random.Random(a.seed)
    data = a.sidd_root / "Data" if (a.sidd_root / "Data").is_dir() else a.sidd_root
    gts = []
    for d in sorted(p for p in data.iterdir() if p.is_dir()):
        g = sorted(d.glob("*GT_SRGB_010*"))          # GT_010 and GT_011 are the same clean image
        if g:
            gts.append((d.name, g[0]))
    if not gts:
        raise SystemExit(f"no *GT_SRGB_010* files under {data}")
    gts = gts[::max(1, a.stride)]
    if a.limit:
        gts = gts[:a.limit]

    scenes = sorted({scene_of(n) for n, _ in gts}); pyr.shuffle(scenes)
    per_scene = {sc: sum(scene_of(n) == sc for n, _ in gts) for sc in scenes}
    print("images per scene:", dict(sorted(per_scene.items())))
    if len(scenes) < 3:
        raise SystemExit(f"Only {len(scenes)} scene(s) found ({scenes}); a scene-level split "
                         "needs at least 3. Paste the folder names and we will pick a different key.")
    val_scenes = set(scenes[:max(1, round(len(scenes) * a.val_frac))])
    print(f"{len(gts)} clean images, {len(scenes)} scenes, val scenes: {sorted(val_scenes)}")

    manifest = {"seed": a.seed, "noise_scale": a.noise_scale, "src": SRC, "out": OUT, "margin": MARGIN,
                "val_scenes": sorted(val_scenes), "bursts": {}}
    for i, (name, path) in enumerate(gts, 1):
        split = "val" if scene_of(name) in val_scenes else "train"
        img = cv2.imread(str(path))                     # BGR, like Role 2
        if img is None or min(img.shape[:2]) < SRC:
            print("  skip", name); continue
        for b in range(a.bursts_per_image):
            y0 = int(rng.integers(0, img.shape[0] - SRC + 1))
            x0 = int(rng.integers(0, img.shape[1] - SRC + 1))
            clean = img[y0:y0 + SRC, x0:x0 + SRC].copy()
            n = int(rng.integers(4, 9))                 # 4-8 frames
            K = float(rng.uniform(15, 60)) / (a.noise_scale ** 2)   # shot-noise std ~ 1/sqrt(K)
            sigma = float(rng.uniform(0.01, 0.03)) * a.noise_scale
            bid = f"{name}_b{b}"
            bdir = a.out / split / bid; bdir.mkdir(parents=True, exist_ok=True)
            (a.out / f"{split}_clean").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(a.out / f"{split}_clean" / f"{bid}.png"), clean)
            Hs, ps = [], []
            for f in range(n):
                H = random_homography(rng, (SRC / 2, SRC / 2))
                clean_view = render(clean, H)
                noisy = add_noise(clean_view, rng, K, sigma)
                cv2.imwrite(str(bdir / f"frame_{f:02d}.png"), noisy)
                Hs.append(H.tolist()); ps.append(round(psnr(noisy, clean_view), 2))
            manifest["bursts"][bid] = {"split": split, "scene": scene_of(name), "source": name,
                                       "crop_xy": [x0, y0], "n": n, "K": K, "sigma": sigma,
                                       "frame_psnr": ps, "homographies": Hs}
        if i % 10 == 0 or i == len(gts):
            print(f"  {i}/{len(gts)}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "manifest.json").write_text(json.dumps(manifest))
    nb = len(manifest["bursts"])
    print(f"wrote {nb} bursts -> {a.out}")


if __name__ == "__main__":
    main()