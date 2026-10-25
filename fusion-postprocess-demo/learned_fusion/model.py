"""
model.py -- learned fusion weighting network (proposal 5.5, Role 3 spec 3.2). TensorFlow / Keras.

For every frame the same small CNN looks at
    [frame (3ch), burst mean (3ch), ransacInlierRatio broadcast to a map (1ch)]
and outputs one logit per pixel. A softmax across the frame axis turns the logits
into per-pixel weights that are positive and sum to 1, and the output is the
weighted sum of the input frames.

Why this shape:
  * output is a convex combination of the input pixels -> it can't invent content,
    so it stays a fusion method and not a hidden denoiser
  * weights are shared across frames -> works for any burst length 1-8; padded
    frames are masked to zero weight
  * the last conv starts at zero -> at step 0 every logit is equal, so the model
    starts out exactly as the naive mean and has to learn to beat it
"""
import tensorflow as tf

IN_CH = 7


class FusionWeightNet(tf.keras.Model):
    def __init__(self, hidden=32, depth=4):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.convs = [tf.keras.layers.Conv2D(hidden, 3, padding="same", activation="relu") for _ in range(depth - 1)]
        self.convs.append(tf.keras.layers.Conv2D(1, 3, padding="same", kernel_initializer="zeros", bias_initializer="zeros"))
        self.config = {"hidden": hidden, "depth": depth}

    def call(self, frames, ratios, mask):
        """frames (B,N,H,W,3) 0-1, ratios (B,N), mask (B,N) bool -> fused (B,H,W,3), weights (B,N,H,W)."""
        shape = tf.shape(frames)
        b, n, h, w = shape[0], shape[1], shape[2], shape[3]
        m = tf.cast(mask, frames.dtype)[:, :, None, None, None]
        mean = tf.reduce_sum(frames * m, 1) / tf.maximum(tf.reduce_sum(m, 1), 1.0)
        conf = tf.broadcast_to(tf.cast(ratios, frames.dtype)[:, :, None, None, None], [b, n, h, w, 1])
        x = tf.concat([frames, tf.broadcast_to(mean[:, None], shape), conf], axis=-1)
        x = tf.reshape(x, [b * n, h, w, IN_CH])
        for conv in self.convs:
            x = conv(x)
        logits = tf.reshape(x, [b, n, h, w])
        logits = tf.where(mask[:, :, None, None], logits, -1e4)
        weights = tf.nn.softmax(logits, axis=1)
        fused = tf.reduce_sum(weights[..., None] * frames, 1)
        return fused, weights
