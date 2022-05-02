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
import sys

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

	#trainX, testX, trainY, testY = train_test_split(
	#	X, Y, test_size = 0.25, random_state = seed, shuffle=True
	#)

	return X, Y, step

def main(plot=False):
	rng = np.random.default_rng(144)
	model = tf.keras.models.load_model('single_output_lstm')
	model.summary()

	X, Y, step = loadData()

	#print(trainX.shape)
	#print(trainX)

	num_assessments = 500
	correct_assessments = 0

	# let's take an arbitrary number of random lines from our input and run
	# them through the decoder!
	for i in range(num_assessments):
		# unsure how to sample from dataset! We need to figure this out.
		input_ind = rng.integers(1, len(X))

		input_seq = X[input_ind:input_ind+1] # we feed this to the model

		ground_truth = X[input_ind:input_ind+2]
		# we can compare the predicted vals to this!
		fullData = list()
		for datum in ground_truth:
			for moment in datum:
				fullData.append(moment[0])
				#print(moment)
		# fullData is now a list of 40 timesteps.
		fullData = np.array(fullData).astype(np.float32)

		#print(input_seq.shape)

		predictions = list()
		while(len(predictions) < (step)):
			prediction = model.predict(input_seq)[0,0]
			predictions.append(prediction)
			newInputSeq = list()
			for tuple in input_seq[0,1:]:
				newInputSeq.append(tuple)
			newInputSeq.append([prediction,0.0])
			input_seq = np.array([newInputSeq]).astype(np.float32)
		predictions = np.array(predictions).astype(np.float32)

		diffInLen = len(fullData)-len(predictions)
		newPredictions = list()
		for j in range(len(fullData)):
			if (i>=diffInLen):
				for p in predictions:
					newPredictions.append(p)
				break
			else:
				newPredictions.append(np.NaN)
		predictions = np.array(newPredictions).astype(np.float32)

		if(plot):
			plt.plot(fullData, color='red',label="Ground truth")
			plt.plot(predictions, color='blue',label="Prediction")

			plt.legend(loc='upper left')
			plt.show()

		#print(input_seq.shape)
		#print(fullData.shape)
		#print(predictions.shape)

		# let's run a simple test: let's figure out of the NN can accurately
		# predict whether the stock will move higher or lower.
		realStockClimbed = (input_seq[0,-1,0] < fullData[-1])
		predictionClimbed = (input_seq[0,-1,0] < predictions[-1])
		if(realStockClimbed==predictionClimbed):
			correct_assessments+=1
		#num_assessments+=1
		if((i%(num_assessments/10))==0):
			print("Iteration", i, "out of", num_assessments)
	print("Our LSTM was correct",
		str((correct_assessments/num_assessments)*100.0) + "% of the time!"
	)

if __name__ == '__main__':
	plotBool = False
	try:
		plotBool = (sys.argv[1].lower() in ['yes', 'plot', 'true'])
	except:
		pass
	print("\n#################################################################")
	print("Displaying plots =", plotBool)
	print("\n#################################################################")
	main(plotBool)