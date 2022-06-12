# Author: Jacob Dawson
#
# This file has ballooned in size. It now initializes and trains the LSTM.
# It imports data from the .json file that should be included in this folder.
# If the .json file does not exist, make sure to call datasetMaker.py.
#
# Needs tensorflow, numpy, and sklearn to be installed.

import tensorflow as tf
import random
import numpy as np
#import math
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from constants import *
from architecture import *

def loadData():
	with open('dataset.json', 'r') as f:
		data = json.load(f)

	X, Y = [], []
	for line in data:
		for i in range(0,len(line)-step-(skip_size+distance_to_predict),skip_size):
			d = i+step
			e = d+distance_to_predict-1
			X.append(line[i:d])
			Y.append(line[e][0])
	X = np.array(X).astype(np.float32)
	Y = np.array(Y).astype(np.float32)

	trainX, testX, trainY, testY = train_test_split(
		X, Y, test_size = 0.3, random_state = seed, shuffle=True,
	)

	return trainX, testX, trainY, testY

# this is the function where our LSTM actually runs
def main():
	trainX, testX, trainY, testY = loadData()
	print('-----------------------------------------------------------------')
	print("Dataset loaded")
	print("Dataset size:", trainX.size)
	print("Dataset shape:", trainX.shape)
	print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
	print("Number of hidden layers:", num_hiddens)
	print('\n-----------------------------------------------------------------')

	'''# this architecture takes in a sequence and guesses what one
	# week ahead will look like.
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
	model.add(tf.keras.layers.Dense(1))'''

	model=createModel()

	model.compile(
		loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
	)
	# we can specify learning rate here!
	# Not sure what the optimal rate is for our problem!
	model.summary()

	path_checkpoint = "model.h5"
	file_checkpoint = Path(path_checkpoint)
	es_callback = tf.keras.callbacks.EarlyStopping(
		monitor="val_loss", min_delta=0, patience=8
	)

	modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
	    monitor="val_loss",
	    filepath=path_checkpoint,
	    verbose=1,
	    save_weights_only=True,
	    save_best_only=True,
	)

	if(file_checkpoint.exists()):
		model.load_weights(path_checkpoint)
		print("Loaded weights from checkpoint", path_checkpoint)
		# if we already have some training done, continue it!
	print("Fitting model.")
	model.fit(
		x=trainX,
		y=trainY,
		epochs=num_epochs,
		batch_size=batch_size,
		callbacks=[es_callback, modelckpt_callback],
		validation_data = (testX, testY)
	)
	print("Model fitted.")

	# two different types of saves:
	model.save("single_output_lstm")
	model.save_weights(path_checkpoint)

if __name__=="__main__":
    random.seed(a=7)
    main()
