# Role 3 -> Role 4 handover: fusion-residual pairs

The `data/` folder is not in git, so Role 4 regenerates the pairs locally. Every step is seeded, so the result
should match Role 3's run.

## Files here
- `synthetic_bursts_manifest.json`: Role 3's `data/synthetic_bursts_fresh/manifest.json` (seed 99, noise_scale 0.3, 69 bursts from 23 SIDD images)
- `fusion_residual_manifest.json`: Role 3's `data/fusion_residual/handover_manifest.json` (SHA-256 + PSNR of every pair)

## Steps (from repo root)
Needs: SIDD Medium sRGB (the full, unchanged `Data` folder: `--stride` counts folders in sorted order),
Role 2's `alignment/` package, `pip install -r fusion-postprocess-demo/requirements.txt`.

```
python fusion-postprocess-demo/learned_fusion/synth_bursts.py --sidd-root <path to SIDD_Medium_Srgb/Data> --out data/synthetic_bursts_fresh --stride 7 --seed 99 --noise-scale 0.3
python fusion-postprocess-demo/learned_fusion/make_fusion_residual_pairs.py --bursts data/synthetic_bursts_fresh --out data/fusion_residual --strategy all
```

Expect `wrote 69 bursts` from the first command and `bursts written: 69 of 69` (57 train, 12 val) from the second.

## Check it matches
Compare `fused_sha256` / `clean_sha256` in your `data/fusion_residual/handover_manifest.json` with
`fusion_residual_manifest.json` here. A different OpenCV version can shift a few hashes. Then compare `psnr_fused_db`
per pair, which should agree within about 0.1 dB.

## Using the pairs
- input: `data/fusion_residual/<split>/<strategy>/<burstId>_fused.png`, target: `data/fusion_residual/<split>/clean/<burstId>_clean.png`
- BGR 8-bit PNG, 32 px already cropped from each side, same coordinates for input and target
- split is by SIDD scene; pass `--val-scenes 004,009` (your own held-out scenes) to the second command to avoid leakage
- sources are SIDD clean images only, no Role 1 held-out test images
