"""
model_builder.py
-----------------
Defines the deep learning architecture used for speech emotion
recognition: a CNN + LSTM hybrid operating on MFCC feature maps.

Input shape: (n_mfcc, time_steps, 1)  -> treated like a 1-channel "image"
"""

from tensorflow.keras import layers, models


def build_cnn_lstm_model(input_shape, num_classes):
    """
    CNN layers learn local spectral-temporal patterns in the MFCCs;
    the LSTM layer then models longer-term temporal dependencies
    across the CNN's output sequence.
    """
    model = models.Sequential()

    # --- Convolutional feature extraction ---
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # --- Reshape CNN output into a sequence for the LSTM ---
    # After the conv/pool stack, shape is (freq', time', channels).
    # We collapse frequency+channel dims and keep the time axis as the
    # sequence dimension for the LSTM.
    shape_after_cnn = model.output_shape  # (batch, freq', time', channels)
    model.add(layers.Permute((2, 1, 3)))  # -> (batch, time', freq', channels)
    model.add(layers.Reshape((shape_after_cnn[2], shape_after_cnn[1] * shape_after_cnn[3])))

    # --- Temporal modeling ---
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(64))
    model.add(layers.Dropout(0.3))

    # --- Classification head ---
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_simple_cnn_model(input_shape, num_classes):
    """
    A lighter, faster-to-train pure-CNN alternative — useful if the
    CNN+LSTM model is too slow on your machine or dataset is small.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
