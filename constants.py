# Any and all constants should be moved here!
# ESPECIALLY if they are shared between files!

# Hyperparameters/things which matter a lot for network construction:
step = 256
batch_size = 8
num_hiddens = 256
lr = 0.0001 # learning rate

# Arbitrary choices:
seed = 2
num_epochs = 100
num_assessments = 56 # controls how many outputs the prediction scripts make
days_predicted = 16 # controls how many days the prediction scripts will project outwards
skip_size = 16
distance_to_predict = 1 # how many days in the future
# is the LSTM trying to predict? 1 means that it will
# predict tomorrow's price, 5 means the price a week from now, etc.
# Effects training!