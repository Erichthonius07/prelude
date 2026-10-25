"""
train.py -- Role 3: train the learned fusion weighting model (strategy 4).

Input: the .npz bursts from build_dataset.py  (<data>/train/*.npz, <data>/val/*.npz).
Loss:  L1 + lambda * (1 - SSIM) against the clean target (proposal 5.5).
Stops early on validation SSIM. Never reads Role 1's held-out test set.

Writes to --out:
  fusion_weightnet.weights.h5 + fusion_weightnet.json   best weights (by val SSIM) + config
  training_log.json       per-epoch train loss / val loss / val SSIM (loss curves for report 7.2)
  gap_report.json         train vs validation SSIM (+ delta) for learned AND hand-crafted strategies,
                          full images, skimage SSIM -- the generalization-gap check (spec 3.2)

Run from repo root:
  python fusion-postprocess-demo/learned_fusion/train.py --data data/fusion_dataset --out checkpoints/learned_fusion
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]

from learned_fusion.model import FusionWeightNet                       # noqa: E402
from learned_fusion.losses import fusion_loss                          # noqa: E402
from learned_fusion.dataset import list_bursts, load_burst, PatchDataset, to_arrays   # noqa: E402
from fusion.module import fuse_arrays                                  # noqa: E402
from common.metrics import computessim, computepsnr                    # noqa: E402


def evaluate_full(model, bursts, lam):
    """Full-image val loss + SSIM with the training loss (used for early stopping)."""
    losses, ssims = [], []
    for frames, ratios, target in bursts:
        f, r, m, t = to_arrays(frames, ratios, target)
        pred, _ = model(f[None], r[None], m[None])
        loss, parts = fusion_loss(pred, t[None], lam)
        losses.append(float(loss)); ssims.append(float(parts["ssim"]))
    return float(np.mean(losses)), float(np.mean(ssims))


def score_split(model, bursts, trimfrac):
    """Per-strategy mean SSIM/PSNR on full images with the same metric code as the ablation."""
    out = {}
    for s in ("naive", "trimmed", "confidence", "learned"):
        ss, ps = [], []
        for frames, ratios, target in bursts:
            fused, _ = fuse_arrays(frames, ratios, s, trimfrac=trimfrac, model=model)
            ss.append(computessim(fused, target)); ps.append(computepsnr(fused, target))
        out[s] = {"ssim": round(float(np.mean(ss)), 4), "psnr": round(float(np.mean(ps)), 3), "n": len(ss)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--patches-per-burst", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.2, help="SSIM weight lambda in L1 + lambda*(1-SSIM)")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--inset", type=int, default=48, help="px cropped from each side (warp borders)")
    ap.add_argument("--patience", type=int, default=6, help="early stop after this many epochs without val SSIM gain")
    ap.add_argument("--trimfrac", type=float, default=0.4, help="for the hand-crafted baselines in gap_report")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N bursts per split")
    a = ap.parse_args()

    tf.keras.utils.set_random_seed(a.seed)
    tr_files, va_files = list_bursts(a.data, "train"), list_bursts(a.data, "val")
    if a.limit:
        tr_files, va_files = tr_files[:a.limit], va_files[:a.limit]
    print(f"train bursts {len(tr_files)}  val bursts {len(va_files)}  GPUs {len(tf.config.list_physical_devices('GPU'))}")

    train_ds = PatchDataset(tr_files, a.patch, a.patches_per_burst, a.inset, augment=True, seed=a.seed)
    val_bursts = [load_burst(f, a.inset) for f in va_files]

    model = FusionWeightNet(a.hidden, a.depth)
    steps = a.epochs * -(-len(train_ds) // a.batch)
    opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(a.lr, decay_steps=steps))

    @tf.function
    def train_step(f, r, m, t):
        with tf.GradientTape() as tape:
            pred, _ = model(f, r, m)
            loss, _ = fusion_loss(pred, t, a.lam)
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss
    a.out.mkdir(parents=True, exist_ok=True)
    ckpt = a.out / "fusion_weightnet.weights.h5"

    val_loss, val_ssim = evaluate_full(model, val_bursts, a.lam)   # epoch 0 = naive mean
    log = [{"epoch": 0, "train_loss": None, "val_loss": val_loss, "val_ssim": val_ssim}]
    print(f"epoch  0  val loss {val_loss:.4f}  val ssim {val_ssim:.4f}  (untrained = naive mean)")
    best, best_epoch, stale = -1.0, 0, 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); tl = []
        for f, r, m, t in train_ds.batches(a.batch):
            tl.append(float(train_step(f, r, m, t)))
        val_loss, val_ssim = evaluate_full(model, val_bursts, a.lam)
        log.append({"epoch": ep, "train_loss": float(np.mean(tl)), "val_loss": val_loss, "val_ssim": val_ssim})
        print(f"epoch {ep:2d}  train loss {np.mean(tl):.4f}  val loss {val_loss:.4f}  val ssim {val_ssim:.4f}  ({time.time() - t0:.0f}s)")
        if val_ssim > best:
            best, best_epoch, stale = val_ssim, ep, 0
            model.save_weights(str(ckpt))
            (a.out / "fusion_weightnet.json").write_text(json.dumps(
                {"config": model.config, "epoch": ep, "val_ssim": val_ssim, "framework": f"tensorflow {tf.__version__}",
                 "args": {k: str(v) for k, v in vars(a).items()}}, indent=1))
        else:
            stale += 1
            if stale >= a.patience:
                print(f"early stop: no val SSIM gain for {a.patience} epochs")
                break
    (a.out / "training_log.json").write_text(json.dumps(log, indent=1))

    # generalization-gap check on full images, best checkpoint, same metric code as the ablation
    from learned_fusion.infer import load_model
    best_model = load_model(ckpt)
    train_bursts = [load_burst(f, a.inset) for f in tr_files]
    tr, va = score_split(best_model, train_bursts, a.trimfrac), score_split(best_model, val_bursts, a.trimfrac)
    gap = {s: {"train_ssim": tr[s]["ssim"], "val_ssim": va[s]["ssim"], "delta": round(tr[s]["ssim"] - va[s]["ssim"], 4),
               "train_psnr": tr[s]["psnr"], "val_psnr": va[s]["psnr"]} for s in tr}
    best_hand = max(("naive", "trimmed", "confidence"), key=lambda s: va[s]["ssim"])
    report = {"best_epoch": best_epoch, "n_train": len(train_bursts), "n_val": len(val_bursts), "inset": a.inset,
              "strategies": gap, "best_handcrafted_on_val": best_hand,
              "learned_minus_best_handcrafted_val_ssim": round(va["learned"]["ssim"] - va[best_hand]["ssim"], 4),
              "note": "delta = train SSIM - val SSIM. A learned delta much larger than the hand-crafted deltas "
                      "means overfitting; report it next to the headline numbers (spec 3.2)."}
    (a.out / "gap_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
