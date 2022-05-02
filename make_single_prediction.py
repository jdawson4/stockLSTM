# the goal of this script is to be able to load our pretrained model
# and see if it can predict actual stock market movements.
# this is where it all comes together!
#
# I'd also like to note that this file takes heavy inspiration from the
# sequence to sequence architecture described by a keras blog post
# describing a network that can translate english to french:
# https://keras.io/examples/nlp/lstm_seq2seq/
# We should cite them in our formal write-ups!

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split

def loadData():
	seed = 7
	rng = np.random.default_rng(seed)
	step = 100
	# we set the step size here.
	# Make sure that this matches the shape given in smallLSTM.py!

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
		X, Y, test_size = 0.25, random_state = seed, shuffle=True
	)

	return trainX, testX, trainY, testY, step

def main():
	rng = np.random.default_rng(3)
	model = tf.keras.models.load_model('single_output_lstm')
	#model.summary()

	trainX, testX, trainY, testY, step = loadData()

	#print(trainX.shape)
	#print(trainX)

	# let's take an arbitrary number of random lines from our input and run
	# them through the decoder!
	for i in range(5):
		# unsure how to sample from dataset! We need to figure this out.
		input_ind = rng.integers(0, len(trainX))

		input_seq = trainX[input_ind:input_ind+1] # we feed this to the model

		ground_truth = trainY[input_ind]
		# we can compare the predicted vals to this!

		#print(input_seq.shape)

		prediction = model.predict(input_seq)[0,0]

		print("On input", input_ind)
		print("Ground truth:",ground_truth)
		print("Prediction:",prediction)
		print('')

		largestPrice = prediction
		smallestPrice = float("Infinity")
		i=0
		for datum in input_seq[0]:
			price = datum[0]
			i+=1
			if price > largestPrice:
				largestPrice = price
			if price < smallestPrice:
				smallestPrice=price
			plt.plot(i,price, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green")
		i+=1
		plt.plot(i,ground_truth, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green",label="Ground truth")
		plt.plot(i,prediction, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="yellow", label="Prediction")

		plt.legend(loc='upper left')
		plt.ylim(smallestPrice-1,largestPrice+1)
		plt.xlim(smallestPrice-1,i+1)
		plt.show()

if __name__ == '__main__':
	main()
