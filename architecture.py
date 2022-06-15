# Author: Jacob Dawson
#
# This is gonna define the architecture used in the smallLSTM training loop
import tensorflow as tf
from tensorflow import keras
from constants import *

# a metric that's unimportant:
def finalDistLoss(y_true, y_pred, sample_weights=None):
	firstTrue = y_true[0][0]
	lastTrue = y_true[-1][0]
	lastPred = y_pred[-1][0]
	wentUp = firstTrue < lastTrue
	predictedUp = firstTrue < lastPred
	return (predictedUp==wentUp)

# Many-to-one architecure:
"""
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
"""
# ^ this is pretty bad!

def createModel():
    input = keras.layers.Input(shape=(step, 2))
    '''en1 = keras.layers.LSTM(num_hiddens,return_sequences=True)(input)
    dr1 = keras.layers.Dropout(0.1)(en1)
    en2 = keras.layers.LSTM(num_hiddens,return_sequences=True)(dr1)
    dr2 = keras.layers.Dropout(0.1)(en2)
    en3 = keras.layers.LSTM(num_hiddens,return_sequences=True)(dr2)
    dr3 = keras.layers.Dropout(0.1)(en3)
    de1 = keras.layers.LSTM(num_hiddens,return_sequences=True)(dr3)
    cat1= keras.layers.Concatenate()([de1,dr3])
    dr4 = keras.layers.Dropout(0.1)(cat1)
    de2 = keras.layers.LSTM(num_hiddens,return_sequences=True)(dr4)
    cat2= keras.layers.Concatenate()([de2,dr2])
    dr5 = keras.layers.Dropout(0.1)(cat2)
    de3 = keras.layers.LSTM(num_hiddens,return_sequences=True)(dr5)
    cat3= keras.layers.Concatenate()([de3,dr1])
    dr6 = keras.layers.Dropout(0.1)(cat3)
    out = keras.layers.LSTM(2,return_sequences=True)(dr6)'''
    lstm1 = keras.layers.LSTM(num_hiddens,return_sequences=True)(input)
    lstm1 = keras.layers.Dropout(0.1)(lstm1)
    lstm2 = keras.layers.LSTM(num_hiddens,return_sequences=True)(lstm1)
    lstm2 = keras.layers.Dropout(0.1)(lstm2)
    lstm2 = keras.layers.Concatenate()([lstm1,lstm2])
    lstm3 = keras.layers.LSTM(num_hiddens,return_sequences=True)(lstm2)
    lstm3 = keras.layers.Dropout(0.1)(lstm3)
    lstm3 = keras.layers.Concatenate()([lstm1,lstm2,lstm3])
    lstm4 = keras.layers.LSTM(num_hiddens,return_sequences=True)(lstm3)
    lstm4 = keras.layers.Dropout(0.1)(lstm4)
    lstm4 = keras.layers.Concatenate()([lstm1,lstm2,lstm3,lstm4])
    out = keras.layers.LSTM(2,return_sequences=True)(lstm4)
    return keras.Model(inputs=input, outputs=out)

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