import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils
import tensorflow.keras.preprocessing.sequence
from scipy.signal import stft

from data_processing import generate_data_sets

# Build this in Keras

# Define our labels that we test against
LABELS = ['yes', 'no', 'up', 'down', 'left', 'right',
          'on', 'off', 'stop', 'go', 'silence', 'unknown']
MAX_SAMPLES = 16000
ID2LABEL = {i: name for i, name in enumerate(LABELS)}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}

def sanitize_spectrogram_params(spectrogram_params):
  if not spectrogram_params:
    spectrogram_params = { 'length': 400, 'step': 160 }
  return spectrogram_params

def get_num_classes():
  return len(LABELS)

def get_input_shape(hparams):
  if hparams['gen_spectrogram']:
    sg_params = sanitize_spectrogram_params(hparams['spectrogram_params'])
    # return the expected shape for the input data
    shape=[0, 0, 0]
    shape[0] = int(np.ceil((MAX_SAMPLES - sg_params['length']) / sg_params['step']))
    shape[1] = (2 ** int(np.ceil(math.log(sg_params['length'],2)))) // 2 + 1
    shape[2] = 2 # This is the phase and amplitude
    return shape
  else:
    return [MAX_SAMPLES]

# Define a Keras sequence object for feeding sequential data
class DataSequence(tensorflow.keras.utils.Sequence):

  def __init__(self,
               dataset,
               hparams,
               shuffle,
               length=MAX_SAMPLES,
               verbose=False):

    # Check expected Y size...    
    assert(len(LABELS) == hparams['num_classes'])
    print ("Building data model...", end='', flush=True)
    if shuffle: np.random.shuffle(dataset)
    self.hparams = hparams
    # The data is stored as a list of WAV samples
    # Create a list of lists of samples, which we'll convert
    # into a single array for data.
    # Turn this into a nparray
    X_list = []
    idx = 0
    self.Y = np.zeros([len(dataset), len(LABELS)])
    for example in dataset:
        X_list.append(example['data'])
        #print(X_list)
        label = example['class']
        if label not in LABELS: label = 'unknown'
        self.Y[idx, LABEL2ID[label]] = 1  # generate one-hot representation
        idx += 1
        if verbose: print(".", end="", flush=True)
    self.X = tensorflow.keras.preprocessing.sequence.pad_sequences(X_list,
                                                                   maxlen=MAX_SAMPLES,
                                                                   dtype='float32',
                                                                   padding='pre',
                                                                   truncating='post')
    if self.hparams['gen_spectrogram']:
      spectrogram_params = sanitize_spectrogram_params(self.hparams['spectrogram_params'])
      
      print("Generating stft for {} examples".format(len(self.X)))
      spectrogram = tf.signal.stft(self.X,
                                   frame_length=spectrogram_params['length'],
                                   frame_step=spectrogram_params['step'],
                                   name='spectrogram')
      phase = tf.angle(spectrogram) / np.pi
      amp = tf.log1p(tf.abs(spectrogram))
      
      x = tf.stack([amp, phase], axis=3)
      x = tf.cast(x, tf.float32)
      self.X = x

    assert(np.shape(x)[1:] == self.hparams['input_shape'])
    print('Done.')
    print("Generated sample array of shape:", np.shape(self.X))
    print("Generated label array of shape:", np.shape(self.Y))

  def input_shape(self):
      return np.shape(self.X)

  def output_shape(self):
      return np.shape(self.Y)
    
  def __len__(self):
    num_samples = int(np.shape(self.X)[0])
    return 5 #int(np.ceil(num_samples / self.hparams['batch_size']))

  def __getitem__(self, idx):
    select = np.array([idx, idx + 1]) * self.hparams['batch_size']
    x = self.X[select[0]:select[1]]
    y = self.Y[select[0]:select[1]]

    return x,y
