import sys
import numpy as np
import tensorflow as tf

import argparse
import datetime

from data_processing import generate_data_sets
from data_processing import keras_data_generator
from data_processing import utils

from keras_models import lstm_model_2d

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
  'seed': 5037,
  'batch_size': 64,
  'keep_prob': 0.7,
  'dropout' : True,
  'batch_norm': True,
  'num_conv_layers': 2,
  'num_lstm_hidden_layers': 2,
  'kernel_size': [3, 3],
  'kernel_stride': [1, 1],
  'lstm_features': 128,
  'conv1d_kernel_size': 3,
  'conv1d_kernel_stride': 1,
  'epochs': 2,
  'gen_spectrogram': True,
  'spectrogram_params': None,
  'loss': 'categorical_crossentropy',
  'optimizer': 'adam',
  'has_gpu': tf.test.is_gpu_available(cuda_only=True),
  'multiprocess': False,
  'threads': 1,
}


# Get the information about the input data shape and the number of classes from
# the data. This will get passed to the model when we generate it.
hparams['input_shape'] = keras_data_generator.get_input_shape(hparams)
hparams['num_classes'] = keras_data_generator.get_num_classes()

#data_generator_verbose = args.verbose
data_generator_verbose = False

# Use the project data processing directives
utils.log_timestamp()
if not args.fast:
  training_data = generate_data_sets.read_dataset(name="training",
                                                  in_dir=args.data_dir)
  training_sequence = keras_data_generator.DataSequence(training_data,
                                                        hparams=hparams,
                                                        shuffle=True,
                                                        verbose=data_generator_verbose)

validation_data = generate_data_sets.read_dataset(name="validation",
                                                  in_dir=args.data_dir)
validation_sequence = keras_data_generator.DataSequence(validation_data,
                                                        hparams=hparams,
                                                        shuffle=False,
                                                        verbose=data_generator_verbose)

test_data = generate_data_sets.read_dataset(name="test",
                                            in_dir=args.data_dir)
test_sequence = keras_data_generator.DataSequence(test_data,
                                                  hparams=hparams,
                                                  shuffle=False,
                                                  verbose=data_generator_verbose)

if args.fast:
  training_sequence = test_sequence

# Create the model and compile
utils.log_timestamp()
model = lstm_model_2d.LstmModel(hparams,
                                model_dir=args.model_dir,
                                verbose=args.verbose)

if args.interactive: exit()

utils.log_timestamp()
print("Starting training....")
history = model.train(training_sequence,
                      validation_sequence,
                      verbose=args.verbose)
print("Done.")

utils.log_timestamp()
print("Running evaluation...")
model.test(test_sequence, verbose=True)
print("Done.")
utils.log_timestamp()
