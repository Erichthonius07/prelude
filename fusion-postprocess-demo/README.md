# fusion-postprocess-demo/ — Role 3

4-way fusion ablation, learned fusion-weighting model, post-processing
(tone mapping + sharpening), and the demo app UI (with offline local caching).
See `Role3_Full_Spec.pdf`. Owned by Role 3.

**Status:** see `docs/STATUS.md` for build status and next action.

## Layout

| Path | What |
|---|---|
| `fusion/adapter.py` | the only place Role 2's AlignedFrameSet is read (ransacInlierRatio only) |
| `fusion/strategies.py` | naive mean, trimmed mean, confidence-weighted mean |
| `fusion/module.py` | `fuse(aligned_result, strategy, pipeline_mode)`: FusionModule entry point, handles `fusion_single` |
| `learned_fusion/model.py, losses.py, dataset.py, train.py, infer.py` | strategy 4: learned per-pixel weighting (TensorFlow / Keras) |
| `learned_fusion/synth_bursts.py, build_dataset.py` | synthetic SIDD bursts -> training .npz |
| `learned_fusion/make_fusion_residual_pairs.py` | fused/clean pairs handed to Role 4 |
| `learned_fusion/diagnose_alignment.py, alignment_experiment.py` | checks on Role 2's alignment |
| `postprocess/` | global + local tone map, unsharp mask, `pipeline.postprocess(img, params)` |
| `pipeline/search.py` | per-burst and multi-image SSIM parameter search, post-processing contribution |
| `pipeline/ablation.py` | 4-way fusion ablation, per-image PSNR/SSIM, Results Service submissions |
| `pipeline/tune_postprocess.py` | picks post-processing params, reports its own SSIM contribution |
| `common/metrics.py, results.py` | PSNR/SSIM, Results Service submission + local-JSON fallback |

## Run order (from repo root, Role 2's `alignment/` package must be importable)

```bash
pip install -r fusion-postprocess-demo/requirements.txt
python -m pytest fusion-postprocess-demo                      # unit tests

# 1. synthetic bursts from SIDD clean images, then aligned training data
python fusion-postprocess-demo/learned_fusion/synth_bursts.py --sidd-root data/sidd/SIDD_Medium_Srgb --out data/synthetic_bursts
python fusion-postprocess-demo/learned_fusion/build_dataset.py --bursts data/synthetic_bursts --out data/fusion_dataset

# 2. train the learned fusion model (writes checkpoint, loss curves, gap_report.json)
python fusion-postprocess-demo/learned_fusion/train.py --data data/fusion_dataset --out checkpoints/learned_fusion

# 3. 4-way ablation (dev numbers on the synthetic val split)
python fusion-postprocess-demo/pipeline/ablation.py --data data/fusion_dataset --split val \
    --checkpoint checkpoints/learned_fusion/fusion_weightnet.weights.h5 --out results/fusion_ablation

# 4. fused/clean pairs for Role 4, then post-processing tuning on them
python fusion-postprocess-demo/learned_fusion/make_fusion_residual_pairs.py --bursts data/synthetic_bursts --out data/fusion_residual --strategy all
python fusion-postprocess-demo/pipeline/tune_postprocess.py --manifest data/fusion_residual/handover_manifest.json \
    --strategy confidence --nrandom 80 --out results/postprocess
```

Add `--results-url http://<laptop>:<port>/<endpoint>` to steps 3 and 4 once Role 1's Results Service is live.
Results are always cached in `results_local/` first.
