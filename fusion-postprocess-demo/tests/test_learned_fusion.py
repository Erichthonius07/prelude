import numpy as np
import tensorflow as tf
from learned_fusion.model import FusionWeightNet
from learned_fusion.losses import ssim, fusion_loss
from learned_fusion.dataset import to_arrays
from learned_fusion.infer import learnedfusion
from fusion.strategies import naivemeanfusion
from common.metrics import computessim


def test_untrained_model_is_naive_mean_and_masks_padding():
    rng = np.random.default_rng(0)
    frames = rng.uniform(0, 1, (5, 16, 16, 3)).astype(np.float32)
    model = FusionWeightNet()
    out, w = learnedfusion(frames, np.linspace(1, 0.5, 5), model, return_weights=True)
    assert np.allclose(out, naivemeanfusion(frames), atol=1e-5)
    assert w.shape == (5, 16, 16) and np.allclose(w.sum(0), 1.0, atol=1e-5)


def test_weights_are_convex_after_random_init():
    tf.keras.utils.set_random_seed(0)
    model = FusionWeightNet()
    f, r, m = to_arrays(np.random.default_rng(1).uniform(0, 1, (3, 12, 12, 3)).astype(np.float32), np.ones(3, np.float32))
    model(f[None], r[None], m[None])
    last = model.convs[-1]
    last.kernel.assign(tf.random.normal(last.kernel.shape, stddev=0.5))
    fused, w = model(f[None], r[None], m[None])
    w = w.numpy()
    assert np.all(w[:, 3:] < 1e-6)                           # padded frames get ~0 weight
    assert np.allclose(w.sum(1), 1.0, atol=1e-5)
    assert fused.numpy().min() >= 0 and fused.numpy().max() <= 1


def test_tf_ssim_matches_skimage_roughly():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1, (48, 48, 3)).astype(np.float32)
    b = np.clip(a + rng.normal(0, 0.05, a.shape), 0, 1).astype(np.float32)
    assert abs(float(ssim(a[None], b[None])) - computessim(a, b)) < 0.03
    assert float(ssim(a[None], a[None])) > 0.999


def test_a_few_steps_reduce_loss_on_ghosted_burst():
    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(3)
    scene = rng.uniform(0.2, 0.8, (32, 32, 3)).astype(np.float32)
    burst = np.stack([np.clip(scene + rng.normal(0, 0.05, scene.shape), 0, 1) for _ in range(5)]).astype(np.float32)
    burst[4] = np.roll(burst[4], 5, axis=0)                  # one ghosted frame, low ratio
    f, r, m, tg = to_arrays(burst, np.array([1, .9, .9, .9, .1], np.float32), scene)
    model = FusionWeightNet(hidden=16, depth=3)
    opt = tf.keras.optimizers.Adam(1e-2)
    first = None
    for _ in range(40):
        with tf.GradientTape() as tape:
            pred, _ = model(f[None], r[None], m[None])
            loss, _ = fusion_loss(pred, tg[None])
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        first = first if first is not None else float(loss)
    assert float(loss) < first
