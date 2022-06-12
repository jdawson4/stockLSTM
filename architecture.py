# Author: Jacob Dawson
#
# This is gonna define the architecture used in the smallLSTM training loop
import tensorflow as tf
from tensorflow import keras
from constants import *

# Many-to-one architecure:

def createModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(
        input_shape = (step, 2),
    )) # defines the inputs! Keep track of that size parameter--important!
    model.add(tf.keras.layers.LSTM(
        num_hiddens,
        return_sequences=True
    ))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(
        num_hiddens//2,
        return_sequences=True
    ))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(
        num_hiddens//4,
        return_sequences=False
    ))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))
    return model

# ^ this is pretty bad!

# What I'm curious about is if the model is being limited by dogshit
# architectural choices rather than the difficulty of modeling the data.
# For instance, is it possible that a sequence-to-sequence architecture
# would work better than this many-to-one situation? What about attention?

if __name__=='__main__':
    model = createModel()
    model.summary()
    '''keras.utils.plot_model(
        model,
        to_file='model_plot.png',
        show_shapes=True,
        show_layer_names=False,
        show_layer_activations=True,
        expand_nested=True
    )''' # this only works on colab???