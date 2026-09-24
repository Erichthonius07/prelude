import itertools
from fusion.strategies import naivemeanfusion, trimmedmeanfusion
from postprocess.tonemapping import localtonemap
from postprocess.sharpening import unsharpmask
from common.metrics import computessim

def runpipeline(burst, fusionfn, tonemapparams, sharpenparams):
    # one full pass: fuse the burst, tone map it, sharpen it
    fused = fusionfn(burst)
    toned = localtonemap(fused, sigma=tonemapparams["sigma"], basegamma=tonemapparams["basegamma"])
    sharpened = unsharpmask(toned, radius=sharpenparams["radius"], amount=sharpenparams["amount"], threshold=sharpenparams["threshold"])
    return sharpened

def gridsearchparams(burst, cleanref, fusionfn):
    # small grid over tonemap+sharpen params, evaluated by ssim against the clean reference
    sigmaoptions = [8.0, 15.0, 25.0]
    basegammaoptions = [0.5, 0.7, 0.9, 1.0]   # 1.0 = identity, lets search choose "dont tone map"
    radiusoptions = [1.0, 1.5, 2.5]
    amountoptions = [0.0, 0.5, 1.0, 1.5]       # 0.0 = identity, lets search choose "dont sharpen"
    thresholdoptions = [0.0, 0.02]

    best = None
    bestscore = -1.0

    combos = itertools.product(sigmaoptions, basegammaoptions, radiusoptions, amountoptions, thresholdoptions)
    for sigma, basegamma, radius, amount, threshold in combos:
        tonemapparams = {"sigma": sigma, "basegamma": basegamma}
        sharpenparams = {"radius": radius, "amount": amount, "threshold": threshold}

        result = runpipeline(burst, fusionfn, tonemapparams, sharpenparams)
        score = computessim(result, cleanref)

        if score > bestscore:
            bestscore = score
            best = {"tonemapparams": tonemapparams, "sharpenparams": sharpenparams, "ssim": score}

    return best