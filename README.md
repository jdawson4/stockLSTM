# stockLSTM
A small, experimental LSTM for predicting stock prices.

Requirements:
1. Tensorflow (best performance requires GPU speedup!)
2. Numpy
3. Sklearn
4. Matplotlib

Files:
1. datasetMaker.py, which creates a file called "dataset.json" with data about stocks
2. smallLSTM.py, which initializes and trains a neural network. It also saves the weights and network to this folder.
3. make_sequence_prediction.py and make_single_prediction.py, which both load the pretrained network and use matplotlib to show some graphs about their predictions.
