# Role 3 status

## Done
- Fusion strategies 1-3 (naive, trimmed, confidence-weighted) and `fusion/module.py` entry point with `pipeline_mode` (`fusion_multi` / `fusion_single`).
- Adapter that reads only `ransacInlierRatio` from Role 2 and skips unaligned frames.
- Strategy 4, learned fusion weighting (TensorFlow / Keras): shared per-frame CNN -> softmax weights across frames,
  starts as the naive mean, L1 + lambda*(1-SSIM) loss, early stopping on val SSIM, train/val SSIM gap report.
- Synthetic burst generator, training dataset builder, fusion-residual pairs for Role 4.
- 4-way ablation runner: per-image PSNR/SSIM, CSV + summary, quick paired bootstrap (uncorrected, dev only).
- Post-processing: global gamma -> local tone map -> unsharp mask; multi-image grid/random search on SSIM;
  separate per-image SSIM contribution.
- Results Service submission format with idempotent `submissionId`, local-JSON fallback, `post_pending()` upload.
- 27 unit tests passing (pytest).

## Not done / blocked on others
- Demo app UI (Android): not started, intentionally left out of this pass.
- Results Service field names in `common/results.py` are a draft. Rename them to Role 1's published contract.
- Official ablation needs Role 1's held-out test set in the same .npz format (frames, ratios, target) plus its datasetId.
- Post-processing should be re-tuned on Role 4/5a's DenoisedFrame once it exists (currently tuned on fused frames).

## Open decisions
- Post-processing search set: spec says the test set; `tune_postprocess.py` defaults to tune on train, report on val. Decide and state it in report 5.4.
- Confirm with Role 2 that the reference frame's `ransacInlierRatio` is 1.0 (fusion picks the reference by argmax ratio).
- RANSAC confidence floor: pass Role 1's calibrated value via `--floor`.
