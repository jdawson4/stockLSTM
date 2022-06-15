# Author: Jacob Dawson
#
# Any and all constants should be moved here!
# ESPECIALLY if they are shared between files!

#from tensorflow.keras.optimizers.schedules import CosineDecay # we might want this for lr!

# Arbitrary choices:
seed = 3
num_epochs = 100
num_assessments = 1000 # controls how many outputs the prediction scripts make
days_predicted = 4 # controls how many days the prediction scripts will project outwards
skip_size = 32 # controls the amount of overlap in the train/test sets (higher skip_size = less overlap)
distance_to_predict = 1 # how many days in the future
# is the LSTM trying to predict? 1 means that it will
# predict tomorrow's price, 5 means the price a week from now, etc.
# Effects training!

# Hyperparameters/things which matter a lot for network construction:
step = 32
batch_size = 256
num_hiddens = 128
initial_lr = 0.1
#lr = CosineDecay(initial_lr, num_epochs) # learning rate as a schedule (didn't seem to work!)
lr = 0.001