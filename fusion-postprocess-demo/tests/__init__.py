import numpy as np
from fusion.strategies import naivemeanfusion, trimmedmeanfusion

def testtrimmedbeatsnaiveonoutlier():
    rng = np.random.default_rng(42)
    scene = rng.uniform(0.3, 0.7, size=(64,64,3)).astype(np.float32)
    burst = np.stack([scene + rng.normal(0,0.05,scene.shape) for i in range(6)]).astype(np.float32)
    burst[3, 20:40, 20:40, :] = 1.0

    naiveerr = np.mean(np.abs(naivemeanfusion(burst)-scene))
    trimmederr = np.mean(np.abs(trimmedmeanfusion(burst, trimfrac=0.4)-scene))

    assert trimmederr < naiveerr

def testnaiveshapeandrange():
    burst = np.random.uniform(0,1,size=(4,16,16,3)).astype(np.float32)
    fused = naivemeanfusion(burst)
    assert fused.shape == (16,16,3)
    assert fused.min() >= 0.0 and fused.max() <= 1.0