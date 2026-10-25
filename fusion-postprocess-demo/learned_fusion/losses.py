"""losses.py -- L = L1 + lambda * (1 - SSIM), the reconstruction loss from proposal 5.5."""
import tensorflow as tf


def ssim(x, y, data_range=1.0):
    """Mean SSIM over (B,H,W,C) tensors, gaussian 11x11 window like Wang et al. 2004."""
    return tf.reduce_mean(tf.image.ssim(x, y, max_val=data_range, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03))


def fusion_loss(pred, target, lam=0.2):
    l1 = tf.reduce_mean(tf.abs(pred - target))
    s = ssim(pred, target)
    return l1 + lam * (1.0 - s), {"l1": l1, "ssim": s}
