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
# I'd also like to note that this file is hideous, easily some of the
# worse-looking code I've written all year.
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
	retData = list()
	for datum in data:
		retData.append(datum[0:1260])

	return np.array(retData, dtype=np.float32)

def main(plot=False):
	rng = np.random.default_rng(seed)
	if(distance_to_predict > 1):
		print("Error: this file does not work on distance_to_predict > 1.")
		print("Please try one of the other prediction files for visualization/testing!")
		print('Alternatively, change the value of distance_to_predict in constants.py, retrain, and retry!')
		return
	data = loadData()
	model = tf.keras.models.load_model('single_output_lstm')
	model.summary()

	correct_assessments = 0

	for i in range(num_assessments):
		# unsure how to sample from dataset! We need to figure this out.
		input_line_num = rng.integers(0, len(data))
		input_line = data[input_line_num]
		
		input_ind = rng.integers(0, len(input_line)-step-days_predicted)

		input_seq = input_line[input_ind:input_ind+step] # we feed this to the model
		expectedPrediction = input_line[input_ind+step:input_ind+step+days_predicted,0]
		input_seq = np.array([input_seq])

		# make days_predicted predictions here
		predictions = list()
		while(len(predictions) < (days_predicted)):
			prediction = model(input_seq)[0,0]
			predictions.append(prediction)
			newInputSeq = list()
			for tuple in input_seq[0,1:]:
				newInputSeq.append(tuple)
			newInputSeq.append([prediction,0.0])
			input_seq = np.array([newInputSeq]).astype(np.float32)
		predictions = np.array(predictions).astype(np.float32)

		# now rephrase the predictions into a plot-able format
		newPredictions = list()
		for j in range(step+days_predicted):
			if (j>=step):
				for p in predictions:
					newPredictions.append(p)
				break
			else:
				newPredictions.append(np.NaN)
		predictions = np.array(newPredictions).astype(np.float32)

		# and do the same for the expected predictions:
		newExpectedPredictions = list()
		for j in range(step+days_predicted):
			if (j>=step):
				for p in expectedPrediction:
					newExpectedPredictions.append(p)
				break
			else:
				newExpectedPredictions.append(np.NaN)
		expectedPrediction = np.array(newExpectedPredictions).astype(np.float32)

		# if we are told to plot, plot
		if(plot):
			plt.plot(input_line[input_ind:input_ind+step,0], color='red',label="Ground Truth")
			plt.plot(expectedPrediction, color = 'green', label="Expected Prediction")
			plt.plot(predictions, color='blue',label="Prediction")

			plt.legend(loc='upper left')
			plt.show()

		# let's run a simple test: let's figure out of the NN can accurately
		# predict whether the stock will move higher or lower.
		realStockClimbed = (input_line[input_ind:input_ind+step,0][-1] < expectedPrediction[-1])
		predictionClimbed = (input_line[input_ind:input_ind+step,0][-1] < predictions[-1])
		if(realStockClimbed==predictionClimbed):
			correct_assessments+=1
		if((i%(num_assessments//10))==0):
			print("Iteration", i, "out of", num_assessments)
	print("Our LSTM was correct",
		str((correct_assessments/num_assessments)*100.0) + "% of the time!"
	)

if __name__ == '__main__':
	print("Intended usage:")
	print("python make_sequence_prediction.py [displayPlots?]")
	plotBool = False
	try:
		plotBool = (sys.argv[1].lower() in ['yes', 'plot', 'true'])
	except:
		pass
	print("\n#################################################################")
	print("Displaying plots =", plotBool)
	print("\n#################################################################")
	main(plotBool)
