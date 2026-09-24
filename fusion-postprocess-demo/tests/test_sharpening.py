import numpy as np
from postprocess.sharpening import unsharpmask
from common.metrics import edgeenergy

def test_sharpeningincreasesedgeenergy():
    rng = np.random.default_rng(7)
    # smooth-ish synthetic image with one hard edge, so theres something to sharpen
    img = np.zeros((64,64,3), dtype=np.float32)
    img[:, 32:, :] = 1.0
    img = img + rng.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    energybefore = edgeenergy(img)
    sharpened = unsharpmask(img, radius=1.5, amount=1.0, threshold=0.0)
    energyafter = edgeenergy(sharpened)

    assert energyafter > energybefore

def test_moreamountmeansmoreedgeenergy():
    rng = np.random.default_rng(7)
    img = np.zeros((64,64,3), dtype=np.float32)
    img[:, 32:, :] = 1.0
    img = img + rng.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    mild = unsharpmask(img, radius=1.5, amount=0.5, threshold=0.0)
    strong = unsharpmask(img, radius=1.5, amount=2.0, threshold=0.0)

    assert edgeenergy(strong) > edgeenergy(mild)

def test_thresholdsuppressesnoiseamplification():
    rng = np.random.default_rng(7)
    # flat image, pure noise, no real edges at all
    flat = np.full((64,64,3), 0.5, dtype=np.float32)
    noisy = flat + rng.normal(0, 0.01, flat.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)

    nothreshold = unsharpmask(noisy, radius=1.5, amount=1.0, threshold=0.0)
    withthreshold = unsharpmask(noisy, radius=1.5, amount=1.0, threshold=0.05)

    # with a threshold above the noise level, the two outputs should end up much closer
    # to the original than sharpening every pixel of noise blindly
    diffnothreshold = np.mean(np.abs(nothreshold - flat))
    diffwiththreshold = np.mean(np.abs(withthreshold - flat))

    assert diffwiththreshold < diffnothreshold