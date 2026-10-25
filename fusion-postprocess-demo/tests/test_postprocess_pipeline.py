import json
import numpy as np
import pytest
from postprocess.pipeline import postprocess, IDENTITY
from pipeline.search import gridsearchpostprocess, postprocesscontribution, candidate_params
from common.results import make_submission, submit


def test_identity_params_are_noop():
    img = np.random.default_rng(0).uniform(0, 1, (32, 32, 3)).astype(np.float32)
    assert np.allclose(postprocess(img, IDENTITY), img, atol=1e-5)


def test_search_over_many_images_beats_identity_on_dark_inputs():
    rng = np.random.default_rng(1)
    refs = [rng.uniform(0.2, 0.8, (32, 32, 3)).astype(np.float32) for _ in range(3)]
    inputs = [np.power(r, 1.25).astype(np.float32) for r in refs]     # too dark, gamma<1 should fix it
    best = gridsearchpostprocess(inputs, refs, nrandom=40, seed=0)
    ident = [row for row in best["table"] if row["params"] == IDENTITY][0]["mean_ssim"]
    assert best["mean_ssim"] >= ident and best["params"]["gamma"] < 1.0
    rows = postprocesscontribution(inputs, refs, best["params"], ["a", "b", "c"])
    assert all(r["ssim_delta"] > 0 for r in rows) and rows[0]["imageId"] == "a"


def test_candidates_include_identity():
    assert IDENTITY in candidate_params(nrandom=5)
    assert len(candidate_params()) >= 4 * 3 * 3 * 3 * 4 * 2


def test_submission_needs_per_image_and_is_cached(tmp_path):
    with pytest.raises(ValueError):
        make_submission("FusionModule", "fusion_naive", [], "dev")
    sub = make_submission("FusionModule", "fusion_naive", [{"imageId": "x", "psnr": 30.0, "ssim": 0.9}], "dev")
    path, uploaded = submit(sub, None, tmp_path)
    assert not uploaded and json.loads(path.read_text())["n"] == 1
