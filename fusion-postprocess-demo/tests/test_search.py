import numpy as np
from pipeline.search import runpipeline, gridsearchparams
from fusion.strategies import naivemeanfusion

def test_runpipeline_shapeandrange():
    rng = np.random.default_rng(3)
    burst = rng.uniform(0,1,size=(6,32,32,3)).astype(np.float32)
    tonemapparams = {"sigma": 15.0, "basegamma": 0.7}
    sharpenparams = {"radius": 1.5, "amount": 1.0, "threshold": 0.02}

    out = runpipeline(burst, naivemeanfusion, tonemapparams, sharpenparams)

    assert out.shape == (32,32,3)
    assert out.min() >= 0.0 and out.max() <= 1.0

def test_gridsearch_findsbetterthandefault():
    rng = np.random.default_rng(3)
    # fake clean scene with a bright region and dark region, gives tonemap something real to do
    clean = np.zeros((32,32,3), dtype=np.float32)
    clean[:, :16, :] = 0.1
    clean[:, 16:, :] = 0.9

    burst = np.stack([clean + rng.normal(0,0.03,clean.shape) for i in range(6)]).astype(np.float32)
    burst = np.clip(burst, 0.0, 1.0)

    best = gridsearchparams(burst, clean, naivemeanfusion)

    # a totally untuned default (mild everything) shouldnt beat the searched best
    defaultresult = runpipeline(burst, naivemeanfusion, {"sigma":15.0,"basegamma":1.0}, {"radius":1.5,"amount":0.0,"threshold":0.0})
    from common.metrics import computessim
    defaultscore = computessim(defaultresult, clean)

    assert best["ssim"] >= defaultscore
    assert best["ssim"] > 0.0