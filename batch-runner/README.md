# batch-runner/ — Role 1

Headless execution of the full 7-variant ablation over the fixed held-out
test set, on the Primary Device. Bypasses live Camera2 capture — loads test
images from storage.

## Owns (per spec 2.5)
- All 7 pipeline variants (single-frame baseline through full pipeline)
- Thermal-throttling handling: enforced spacing between runs, thermal state logged per submission
- Uploads results to the Results Service in bulk

_Not yet implemented — depends on the data contract and a working pipeline._
