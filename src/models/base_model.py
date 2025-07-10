import tensorflow as tf
from tensorflow.keras import layers, models

from config import model_config as MC
from src.utils.metrics import F1Score

__all__ = ["build_base_model"]


def _core(inputs, attention: bool):
    x = layers.Conv1D(MC.MODEL["conv_filters"], MC.MODEL["kernel_size"], activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(MC.MODEL["dropout_conv"])(x)
    x = layers.LSTM(MC.MODEL["lstm_units_1"], return_sequences=True)(x)
    x = layers.Dropout(MC.MODEL["dropout_lstm1"])(x)
    x = layers.LSTM(MC.MODEL["lstm_units_2"], return_sequences=attention)(x)
    return x


def build_base_model(input_shape, loss="binary_crossentropy", optimizer=None, attention=False):
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(model_config.LEARNING_RATE, clipnorm=model_config.CLIPNORM)

    inputs = tf.keras.Input(shape=input_shape)
    x = _core(inputs, attention)

    if attention:
        from src.models.cnn_lstm import AttentionLayer  # local import to avoid cycles
        x = AttentionLayer()(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(MC.MODEL["dropout_dense"])(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score()])
    return model