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

# ---- multi-image post-processing search (spec 3.3) -------------------------------
# gridsearchparams above tunes on ONE burst. The real search has to pick one parameter
# set for MANY images, scored by mean SSIM, and on post-processing alone (its input is
# the DenoisedFrame, or the fused frame until Role 4/5a's denoiser is wired in).
import random
import numpy as np
from postprocess.pipeline import postprocess, IDENTITY

POSTPROCESS_GRID = {
    "gamma": [0.8, 0.9, 1.0, 1.1],
    "sigma": [8.0, 15.0, 25.0],
    "basegamma": [0.7, 0.85, 1.0],
    "radius": [1.0, 1.5, 2.5],
    "amount": [0.0, 0.25, 0.5, 1.0],     # kept mild, over-sharpening re-amplifies noise
    "threshold": [0.0, 0.02],
}


def candidate_params(grid=None, nrandom=0, seed=0):
    # full grid when nrandom=0, otherwise nrandom random picks (identity always included)
    grid = grid or POSTPROCESS_GRID
    keys = list(grid)
    if nrandom <= 0:
        combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    else:
        rng = random.Random(seed)
        combos = [{k: rng.choice(grid[k]) for k in keys} for _ in range(nrandom)]
    if IDENTITY not in combos:
        combos.insert(0, dict(IDENTITY))
    return combos


def gridsearchpostprocess(inputs, refs, grid=None, nrandom=0, seed=0):
    # inputs/refs: lists of HWC float32 0-1 images. returns best params, its mean ssim, and the full table
    if len(inputs) != len(refs) or not inputs:
        raise ValueError("need matching, non-empty lists of inputs and references")
    table = []
    for params in candidate_params(grid, nrandom, seed):
        score = float(np.mean([computessim(postprocess(x, params), r) for x, r in zip(inputs, refs)]))
        table.append({"params": params, "mean_ssim": score})
    table.sort(key=lambda row: -row["mean_ssim"])
    return {"params": table[0]["params"], "mean_ssim": table[0]["mean_ssim"], "table": table}


def postprocesscontribution(inputs, refs, params, ids=None):
    # per-image ssim before vs after post-processing, so its value isn't buried in the full-pipeline number
    ids = ids or [str(i) for i in range(len(inputs))]
    rows = []
    for i, x, r in zip(ids, inputs, refs):
        before, after = computessim(x, r), computessim(postprocess(x, params), r)
        rows.append({"imageId": i, "ssim_before": before, "ssim": after, "ssim_delta": after - before})
    return rows
