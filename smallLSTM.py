# This file has ballooned in size. It now initializes and trains the LSTM.
# It imports data from the .json file that should be included in this folder.
# If the .json file does not exist, make sure to call datasetMaker.py.
#
# Needs tensorflow, numpy, and sklearn to be installed.

import tensorflow as tf
import random
import numpy as np
import math
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

def loadData():
	seed = 7
	rng = np.random.default_rng(seed)
	step = 100 # we set the step size here.

	with open('dataset.json', 'r') as f:
		data = json.load(f)

	X, Y = [], []
	for line in data:
		for i in range(0,len(line)-step-step, step):
			d = i+step
			#print(line[i:d])
			X.append(line[i:d])
			Y.append(line[d][0])
	X = np.array(X).astype(np.float32)
	Y = np.array(Y).astype(np.float32)

	trainX, testX, trainY, testY = train_test_split(
		X, Y, test_size = 0.3, random_state = seed, shuffle=True,
	)

	return trainX, testX, trainY, testY, step

# this is the function where our LSTM actually runs
def main():
	num_hiddens = 256 # let's declare the number up here!
	num_epochs = 100 # and declare the epochs up here!
	trainX, testX, trainY, testY, step = loadData()
	print('-----------------------------------------------------------------')
	print("Dataset loaded")
	print("Dataset size:", trainX.size)
	print("Dataset shape:", trainX.shape)
	print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
	print("Number of hidden layers:", num_hiddens)
	print('\n-----------------------------------------------------------------')

	# this architecture takes in a sequence and guesses what one
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
	model.add(tf.keras.layers.Dense(1))

	model.compile(
		loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
	)
	# we can specify learning rate here!
	# Not sure what the optimal rate is for our problem!
	model.summary()

	path_checkpoint = "model2.h5"
	file_checkpoint = Path(path_checkpoint)
	es_callback = tf.keras.callbacks.EarlyStopping(
		monitor="val_loss", min_delta=0, patience=7
	)

	modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
	    monitor="val_loss",
	    filepath=path_checkpoint,
	    verbose=1,
	    save_weights_only=True,
	    save_best_only=True,
	)

	#print(trainY.shape)
	#print(trainY[0].shape)
	if(file_checkpoint.exists()):
		model.load_weights(path_checkpoint)
		print("Loaded weights from checkpoint", path_checkpoint)
		# if we already have some training done, continue it!
	print("Fitting model.")
	model.fit(
		x=trainX,
		y=trainY,
		epochs=num_epochs,
		batch_size=8,
		callbacks=[es_callback, modelckpt_callback],
		validation_data = (testX, testY)
	)
	print("Model fitted.")

	model.save("single_output_lstm")
	model.save_weights(path_checkpoint)

	# and maybe this code will work? I'm gonna go on a walk now,
	# good luck little LSTM!

	print("Predicting and evaluating model.")
	#trainPredict = model.predict(trainX)
	#testPredict= model.predict(testX)
	#print(testPredict)
	#predicted=np.concatenate((trainPredict,testPredict),axis=0)

	#trainScore = model.evaluate(trainX, trainY, verbose=0)
	testScore = model.evaluate(testX, testY, verbose=0)
	print("Score:")
	print(testScore)

if __name__=="__main__":
    random.seed(a=7)
    main()