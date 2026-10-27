# training/model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_dncnn(num_layers: int = 17, features: int = 64, channels: int = 3) -> tf.keras.Model:
    """
    Builds the resolution-agnostic DnCNN model using residual learning.
    Default: 17 layers, 64 channels (~557k parameters).
    Still-capture variant: num_layers=9, features=32 (~66k parameters).
    """
    inputs = layers.Input(shape=(None, None, channels), name="input")

    # Layer 1: Conv + ReLU (no BatchNorm on the first layer)
    x = layers.Conv2D(
        filters=features,
        kernel_size=(3, 3),
        padding="same",
        use_bias=True,
        kernel_initializer="he_normal",
        name="conv_1"
    )(inputs)
    x = layers.ReLU(name="relu_1")(x)

    # Layers 2 to (num_layers - 1): Conv + BatchNorm + ReLU
    for i in range(2, num_layers):
        x = layers.Conv2D(
            filters=features,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=f"conv_{i}"
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.ReLU(name=f"relu_{i}")(x)

    # Final Layer: Predict the 3-channel residual noise map
    predicted_noise = layers.Conv2D(
        filters=channels,
        kernel_size=(3, 3),
        padding="same",
        use_bias=True,
        kernel_initializer="he_normal",
        name="predicted_noise"
    )(x)

    # Residual subtraction: Clean Image = Noisy Input - Predicted Noise
    denoised_output = layers.Subtract(name="output")([inputs, predicted_noise])

    return models.Model(inputs=inputs, outputs=denoised_output, name=f"DnCNN_{num_layers}L")

if __name__ == "__main__":
    model = build_dncnn()
    model.summary()