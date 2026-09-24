import numpy as np
from postprocess.tonemapping import globaltonemap, localtonemap
from common.metrics import edgeenergy

def test_globaltonemap_identitygamma():
    # gamma=1 should be a no-op, pow(x,1)==x
    img = np.random.uniform(0,1,size=(32,32,3)).astype(np.float32)
    out = globaltonemap(img, gamma=1.0)
    assert np.allclose(out, img, atol=1e-6)

def test_globaltonemap_lowgammabrightens():
    # gamma<1 should lift midtones, mean brightness should go up
    img = np.full((32,32,3), 0.25, dtype=np.float32)
    out = globaltonemap(img, gamma=0.5)
    assert out.mean() > img.mean()

def test_globaltonemap_highgammadarkens():
    # gamma>1 should push midtones down, mean brightness should go down
    img = np.full((32,32,3), 0.5, dtype=np.float32)
    out = globaltonemap(img, gamma=1.8)
    assert out.mean() < img.mean()

def test_localtonemap_compressesglobalcontrast():
    # bright region and dark region, big gap between them
    # local tone map should shrink that gap since it squashes the base layer
    img = np.zeros((64,64,3), dtype=np.float32)
    img[:, :32, :] = 0.05   # dark half
    img[:, 32:, :] = 0.95   # bright half

    gapbefore = img[:, 32:, :].mean() - img[:, :32, :].mean()

    out = localtonemap(img, sigma=15.0, basegamma=0.6)
    gapafter = out[:, 32:, :].mean() - out[:, :32, :].mean()

    assert gapafter < gapbefore

def test_localtonemap_keepsdetailnotjustblurred():
    # fine stripes inside one region, should survive local tone mapping
    # since detail layer gets added back untouched
    img = np.full((64,64,3), 0.5, dtype=np.float32)
    img[:, ::2, :] = 0.7   # alternating stripes, high freq detail

    edgebefore = edgeenergy(img)
    tonemapped = localtonemap(img, sigma=15.0, basegamma=0.6)
    edgeaftertonemap = edgeenergy(tonemapped)

    # a plain blur wouldve wrecked the stripes, tone mapped version should keep most of that edge energy
    assert edgeaftertonemap > edgebefore * 0.7