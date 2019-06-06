import sys
import argparse
import numpy as np
import tensorflow as tf

from data_processing import generate_data_sets
from data_processing import keras_data_generator
from data_processing import utils

from keras_models import baseline

print("Using python version:", sys.version)
print("Using numpy version:", np.__version__)
print("Using TensorFlow version:", tf.__version__)


utils.log_timestamp()

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
parser.add_argument("-i", "--interactive", help="Load data and models and exit", action="store_true")
parser.add_argument("-f", "--fast", help="Run validation vs test (no training data processed)", action="store_true")
parser.add_argument("-data_dir", help="Input data location", default="./data")
parser.add_argument("-model_dir", help="Output data file location", default="./model")
args = parser.parse_args()

hparams = {
    'batch_size': 64,
    #'keep_prob': 0.5,
    #'batch_norm': True,
    'use_conv2d': True,
    'keep_prob': 1.0,
    'batch_norm': False,
    'epochs': 10,
    'gen_spectrogram': True,
    'spectrogram_params': None,
    'loss': 'categorical_crossentropy',
    'optimizer': 'adam',
    'multiprocess': False,
    'threads': 1
}


# Get the information about the input data shape and the number of classes from
# the data. This will get passed to the model when we generate it.
hparams['input_shape'] = keras_data_generator.get_input_shape(hparams)
hparams['num_classes'] = keras_data_generator.get_num_classes()

# Use the project data processing directives
if not args.fast:
  training_data = generate_data_sets.read_dataset(name="training",
                                                  in_dir=args.data_dir)
  training_sequence = keras_data_generator.DataSequence(training_data,
                                                        hparams=hparams,
                                                        shuffle=True)

validation_data = generate_data_sets.read_dataset(name="validation",
                                                  in_dir=args.data_dir)
validation_sequence = keras_data_generator.DataSequence(validation_data,
                                                        hparams=hparams,
                                                        shuffle=False)

test_data = generate_data_sets.read_dataset(name="test",
                                            in_dir=args.data_dir)
test_sequence = keras_data_generator.DataSequence(test_data,
                                                  hparams=hparams,
                                                  shuffle=False)

if args.fast:
  training_sequence = test_sequence

# Create the model and compile
utils.log_timestamp()
model = baseline.BaselineModel(hparams,
                               model_dir=args.model_dir,
                               verbose=args.verbose)

if args.interactive: exit()

utils.log_timestamp()
print("Starting training....")
history = model.train(training_sequence,
                      validation_sequence,
                      verbose=True)
print("Done.")

utils.log_timestamp()
print("Running evaluation...")
model.test(test_sequence, verbose=True)
print("Done.")
utils.log_timestamp()
