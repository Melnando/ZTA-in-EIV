import tensorflow as tf
from tensorflow.keras import backend as K

__all__ = ["F1Score", "FocalLoss"]

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p, r = self.precision.result(), self.recall.result()
        return 2 * (p * r) / (p + r + K.epsilon())

    def reset_state(self):
        self.precision.reset_state(); self.recall.reset_state()


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        ce = -y_true * K.log(y_pred)
        return tf.reduce_mean(self.alpha * tf.pow(1 - y_pred, self.gamma) * ce, axis=-1)