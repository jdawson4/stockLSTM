# Author: Jacob Dawson
#
# The goal of this file is just to use a built-in keras evaluation.
# No plotting, no projecting outwards, just a straight-up accuracy score.

import numpy as np
import tensorflow as tf
import json
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

def main():
    X, Y = loadData()
    model = tf.keras.models.load_model('single_output_lstm')
    model.summary()
    model.compile(
        metrics = [tf.metrics.MeanAbsolutePercentageError()],
        loss = 'mse',
    ) # don't worry, we're not saving this anywhere.
    scores = model.evaluate(
        x=X,
        y=Y,
        batch_size = batch_size,
        steps=step
    )
    print("Note that loss is Mean Squared Error")
    for i in range(len(scores)):
        score = scores[i]
        metricName = model.metrics_names[i]
        print(metricName, "achieved:", round(score,3))

if __name__ == '__main__':
	main()