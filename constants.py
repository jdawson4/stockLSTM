# Any and all constants should be moved here!
# ESPECIALLY if they are shared between files!

#from tensorflow.keras.optimizers.schedules import CosineDecay # we might want this for lr!

# Arbitrary choices:
seed = 4
num_epochs = 100
num_assessments = 512 # controls how many outputs the prediction scripts make
days_predicted = 16 # controls how many days the prediction scripts will project outwards
skip_size = 16 # controls the amount of overlap in the train/test sets (higher skip_size = less overlap)
distance_to_predict = 5 # how many days in the future
# is the LSTM trying to predict? 1 means that it will
# predict tomorrow's price, 5 means the price a week from now, etc.
# Effects training!

# Hyperparameters/things which matter a lot for network construction:
step = 256
batch_size = 8
num_hiddens = 256
initial_lr = 0.1
#lr = CosineDecay(initial_lr, num_epochs) # learning rate as a schedule (didn't seem to work!)
lr = 0.001