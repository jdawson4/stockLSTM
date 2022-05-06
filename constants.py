# Any and all constants should be moved here!
# ESPECIALLY if they are shared between files!

# Hyperparameters/things which matter a lot for network construction:
step = 256
batch_size = 8
num_hiddens = 256
lr = 0.0001 # learning rate

# Arbitrary choices:
seed = 3
num_epochs = 100
num_assessments = 50 # controls how many outputs the prediction scripts make
days_predicted = 10
skip_size = 16