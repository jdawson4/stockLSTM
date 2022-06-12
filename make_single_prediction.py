# Author: Jacob Dawson
#
# the goal of this script is to be able to load our pretrained model
# and see if it can predict actual stock market movements.
# this is where it all comes together!
#
# I'd also like to note that this file takes heavy inspiration from the
# sequence to sequence architecture described by a keras blog post
# describing a network that can translate english to french:
# https://keras.io/examples/nlp/lstm_seq2seq/
# We should cite them in our formal write-ups!
#
# This file ONLY works for many-to-one architecture!

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
		for i in range(0,len(line)-step-(skip_size+distance_to_predict),skip_size):
			d = i+step
			e = d+distance_to_predict-1
			X.append(line[i:d])
			Y.append(line[e][0])
	X = np.array(X).astype(np.float32)
	Y = np.array(Y).astype(np.float32)

	return X, Y

def main(plot=False):
	X, Y = loadData()
	rng = np.random.default_rng(seed)
	#num_assessments = 5
	model = tf.keras.models.load_model('single_output_lstm')
	model.summary()
	correct_assessments = 0

	# let's take an arbitrary number of random lines from our input and run
	# them through the decoder!
	for i in range(num_assessments):
		# unsure how to sample from dataset! We need to figure this out.
		input_ind = rng.integers(0, len(X))

		input_seq = X[input_ind:input_ind+1] # we feed this to the model
		# (I realize that this is a strange shape but trust me we need it like this)

		ground_truth = Y[input_ind]
		# we can compare the predicted vals to this!

		prediction = model.predict(input_seq)[0,0]

		#print("On input", input_ind)
		#print("Ground truth:",ground_truth)
		#print("Prediction:",prediction)
		#print('')

		if(plot):
			j=0
			groundTruthXToPlot = list()
			groundTruthYToPlot = list()
			for datum in input_seq[0]:
				price = datum[0]
				j+=1
				groundTruthYToPlot.append(price)
				groundTruthXToPlot.append(j)
			plt.plot(j+distance_to_predict,ground_truth, marker="x", markersize=5, markeredgecolor="blue", markerfacecolor="blue", label="Prediction Target")
			plt.plot(groundTruthXToPlot, groundTruthYToPlot, label='Ground Truth',color='blue')
			plt.plot(j+distance_to_predict,prediction, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="yellow", label="LSTM Prediction")
			plt.legend(loc='upper left')
			plt.show()
		realStockClimbed = (input_seq[0,-1,0] < ground_truth)
		predictionClimbed = (input_seq[0,-1,0] < prediction)
		if(realStockClimbed==predictionClimbed):
			correct_assessments+=1
	print("Performed", num_assessments,"tests.")
	print("Our LSTM was correct",
		str((correct_assessments/num_assessments)*100.0) + "% of the time!"
	)

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
