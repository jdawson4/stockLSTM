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
import matplotlib.pyplot as plt
import json
import sys
from constants import *

def loadData():
	with open('dataset.json', 'r') as f:
		data = json.load(f)

	X, Y = [], []
	for line in data:
		for i in range(0,len(line)-step-step, skip_size):
			d = i+step
			X.append(line[i:d])
			Y.append(line[d][0])
	X = np.array(X).astype(np.float32)
	Y = np.array(Y).astype(np.float32)

	return X, Y

def main(plot=False):
	X, Y = loadData()
	rng = np.random.default_rng(seed)
	#num_assessments = 5
	model = tf.keras.models.load_model('single_output_lstm')
	model.summary()

	# let's take an arbitrary number of random lines from our input and run
	# them through the decoder!
	for i in range(num_assessments):
		# unsure how to sample from dataset! We need to figure this out.
		input_ind = rng.integers(0, len(X))

		input_seq = X[input_ind:input_ind+1] # we feed this to the model

		ground_truth = X[input_ind]
		# we can compare the predicted vals to this!

		prediction = model.predict(input_seq)[0,0]

		print("On input", input_ind)
		print("Ground truth:",ground_truth[-1])
		print("Prediction:",prediction)
		print('')

		if(plot):
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
			plt.plot(i,ground_truth[-1,0], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green",label="Ground truth")
			plt.plot(i,prediction, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="yellow", label="Prediction")

			plt.legend(loc='upper left')
			plt.ylim(smallestPrice-1,largestPrice+1)
			plt.xlim(smallestPrice-1,i+1)
			plt.show()

if __name__ == '__main__':
	print("Intended usage:")
	print("python make_single_prediction.py [displayPlots?]")
	plotBool = False
	try:
		plotBool = (sys.argv[1].lower() in ['yes', 'plot', 'true'])
	except:
		pass
	print("\n#################################################################")
	print("Displaying plots =", plotBool)
	print("\n#################################################################")
	main(plotBool)
