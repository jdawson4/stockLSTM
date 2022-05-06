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
	if(distance_to_predict > 1):
		print("Error: this file does not work on distance_to_predict > 1.")
		print("Please try one of the other prediction files for visualization/testing!")
		print('Alternatively, change the value of distance_to_predict in constants.py, retrain, and retry!')
		return
	X, Y = loadData()
	rng = np.random.default_rng(seed)
	model = tf.keras.models.load_model('single_output_lstm')
	model.summary()

	correct_assessments = 0

	for i in range(num_assessments):
		# unsure how to sample from dataset! We need to figure this out.
		input_ind = rng.integers(1, len(X))

		input_seq = X[input_ind:input_ind+1] # we feed this to the model

		ground_truth = X[input_ind:input_ind+2]
		# we can compare the predicted vals to this!
		fullData = list()
		done=False
		for datum in ground_truth:
			for moment in datum:
				fullData.append(moment[0])
				if(len(fullData) > (step+num_assessments)):
					done=True
					break
			if(done):
				break
		# fullData is now a list of [step_size+num_assessments] timesteps.
		fullData = np.array(fullData).astype(np.float32)

		predictions = list()
		while(len(predictions) < (days_predicted)):
			prediction = model.predict(input_seq)[0,0]
			predictions.append(prediction)
			newInputSeq = list()
			for tuple in input_seq[0,1:]:
				newInputSeq.append(tuple)
			newInputSeq.append([prediction,0.0])
			input_seq = np.array([newInputSeq]).astype(np.float32)
		predictions = np.array(predictions).astype(np.float32)

		newPredictions = list()
		for j in range(step+days_predicted):
			if (j>=step):
				for p in predictions:
					newPredictions.append(p)
				break
			else:
				newPredictions.append(np.NaN)
		predictions = np.array(newPredictions).astype(np.float32)

		if(plot):
			plt.plot(fullData[:step+days_predicted], color='red',label="Ground truth")
			plt.plot(predictions, color='blue',label="Prediction")

			plt.legend(loc='upper left')
			plt.show()

		# let's run a simple test: let's figure out of the NN can accurately
		# predict whether the stock will move higher or lower.
		realStockClimbed = (input_seq[0,-1,0] < fullData[-1])
		predictionClimbed = (input_seq[0,-1,0] < predictions[-1])
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
