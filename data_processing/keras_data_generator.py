import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils
import tensorflow.keras.preprocessing.sequence
from scipy.signal import spectrogram

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
    spectrogram_params = { 'window_length': 512,
                           'window_step': 160,
                           'fft_length': 512}
  # Determine any second order parameters we need
  if not spectrogram_params['fft_length']:
    # We need to compute the power 2 length (pad as required)
    spectrograam_params['fft_length'] = (2 ** int(no.ceil(math.log(spectrogram_params['window_length'],2))))

  print('Spectrogram Parameters: ', spectrogram_params)
  return spectrogram_params

def get_num_classes():
  return len(LABELS)

def get_input_shape(hparams):
  if hparams['gen_spectrogram']:
    sg_params = sanitize_spectrogram_params(hparams['spectrogram_params'])
    # return the expected shape for the input data
    shape=[0, 0]
    # shape[0] is the result of the computation (one sided)
    shape[0] = sg_params['fft_length'] // 2 + 1
    # shape[1] is the number of segments we compute
    shape[1] = int(np.ceil((MAX_SAMPLES - sg_params['window_length'] + 1) / sg_params['window_step']))
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
    self.dataset = dataset
    self.hparams = hparams
    self.verbose = verbose

    # Compute the spectrogram parameters
    self.spectrogram_params = sanitize_spectrogram_params(self.hparams['spectrogram_params'])

    # The data is stored as a list of WAV samples
    # Create a list of lists of samples, which we'll convert
    # into a single array for data.
    # Turn this into a nparray
    X_list = []
    idx = 0
    self.flist = []
    self.Y = np.zeros([len(self.dataset), len(LABELS)])
    for example in self.dataset:
        X_list.append(example['data'])
        self.flist.append(example['filename'])
        label = example['class']
        if label not in LABELS: label = 'unknown'
        self.Y[idx, LABEL2ID[label]] = 1  # generate one-hot representation
        idx += 1
        if (idx % 100) and verbose: print(".", end="", flush=True)
    self.X = tensorflow.keras.preprocessing.sequence.pad_sequences(X_list,
                                                                   maxlen=MAX_SAMPLES,
                                                                   dtype='float32',
                                                                   padding='pre',
                                                                   truncating='post')
    print('Done.')
    if self.verbose:
      print("Generated sample array of shape:", np.shape(self.X))
      print("Generated label array of shape:", np.shape(self.Y))

  def input_shape(self):
      return np.shape(self.X)

  def output_shape(self):
      return np.shape(self.Y)
    
  def __len__(self):
    num_samples = int(np.shape(self.X)[0])
    return int(np.ceil(num_samples / self.hparams['batch_size']))

  def __getitem__(self, idx):
    select = np.array([idx, idx + 1]) * self.hparams['batch_size']
    x = self.X[select[0]:select[1]]

    if self.hparams['gen_spectrogram']:
      
      if self.verbose: print("Generating spectrogram for {} examples".format(len(x)))
      p = self.spectrogram_params
      f, t, s = spectrogram(x,
                            nperseg=p['window_length'],
                            noverlap=p['window_length'] - p['window_step'],
                            scaling='spectrum',
                            mode='magnitude',
      )
      x = s

    if self.verbose:
      print (np.shape(x)[1:])
      print (self.hparams['input_shape'])

    #assert(np.shape(x)[1:] == self.hparams['input_shape'])
    y = self.Y[select[0]:select[1]]

    return x,y

  def get_data_tuple(self, idx):
    x = self.X[idx]
    f = self.flist[idx]
    y = self.Y[idx]
    return x, f, y
