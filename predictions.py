# Author: Jacob Dawson
#
# Note: this file ONLY works for the many-to-many architecture!

from venv import create
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import sys
from constants import *
from architecture import *

def loadData():
	with open('dataset.json', 'r') as f:
		data = json.load(f)

	X, Y = [], []
	for line in data:
		for i in range(0,len(line)-(2*step),skip_size):
			d = i+step
			e = d+step
			X.append(line[i:d])
			Y.append(line[d:e])
	X = np.array(X).astype(np.float32)
	Y = np.array(Y).astype(np.float32)

	return X,Y

def main(plot=False):
	rng = np.random.default_rng(seed)

	X, Y = loadData()
	#model = tf.keras.models.load_model('stock_lstm')
	model = createModel()
	model.load_weights('model.h5')
	model.summary()

	correct_assessments = 0

	for i in range(num_assessments):
		selection = rng.integers(0, len(X))
		x = X[selection]
		y_true = Y[selection]
		y_pred = model(tf.expand_dims(x,0))[0]


		x_plot = list()
		for price, vol in x:
			x_plot.append(price)
		y_true_plot = list()
		for price, vol in y_true:
			y_true_plot.append(price)
		y_pred_plot = list()
		for price, vol in y_pred:
			y_pred_plot.append(price)
		
		if plot:
			plt.plot(y_true_plot, color='red',label="Ground Truth")
			plt.plot(y_pred_plot, color='blue',label="Prediction")
			plt.legend(loc='upper left')
			plt.show()

		# determine if the lstm predicted in the correct direction:
		realStockClimbed = (x_plot[-1] < y_true_plot[-1])
		predictionClimbed = (x_plot[-1] < y_pred_plot[-1])
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
