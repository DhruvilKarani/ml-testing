import tensorflow as tf
import tempfile
import os
import numpy as np
print('TensorFlow version: {}'.format(tf.__version__))



class SimpleMLPClassifier(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_layer = tf.keras.layers.Dense(self.hidden_dim, activation='relu', input_shape=(self.input_dim,), name='input_layer')
        self.hidden_layer = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='hidden_layer')
        self.output_layer = tf.keras.layers.Dense(self.output_dim, activation='softmax', name='output_layer')


    def call(self, x):
        output = self.input_layer(x)
        output = self.hidden_layer(output)
        output = self.output_layer(output)
        return output

 
    