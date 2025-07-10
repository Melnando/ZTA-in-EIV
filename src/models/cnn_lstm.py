import tensorflow as tf
from tensorflow.keras import backend as K

__all__ = ["AttentionLayer"]

class AttentionLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight("W", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform")
        self.b = self.add_weight("b", shape=(input_shape[-1],), initializer="zeros")
        self.u = self.add_weight("u", shape=(input_shape[-1], 1), initializer="glorot_uniform")

    def call(self, x):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.squeeze(K.dot(uit, self.u), -1)
        a = K.softmax(ait)
        return K.sum(x * K.expand_dims(a, -1), axis=1)