"""
pipeline.py -- PostProcessModule: DenoisedFrame -> FinalImage.

Order (spec 3.3): global tone map -> local tone map -> unsharp mask.
Every parameter has an identity setting (gamma 1, basegamma 1, amount 0) so the
search can decide a step shouldn't run at all.
"""
from postprocess.tonemapping import globaltonemap, localtonemap
from postprocess.sharpening import unsharpmask

IDENTITY = {"gamma": 1.0, "sigma": 15.0, "basegamma": 1.0, "radius": 1.5, "amount": 0.0, "threshold": 0.0}


def postprocess(image, params=None):
    p = dict(IDENTITY, **(params or {}))
    out = globaltonemap(image, gamma=p["gamma"])
    out = localtonemap(out, sigma=p["sigma"], basegamma=p["basegamma"])
    return unsharpmask(out, radius=p["radius"], amount=p["amount"], threshold=p["threshold"])
