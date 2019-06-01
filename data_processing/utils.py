import datetime

import matplotlib.pyplot as plt

def log_timestamp():
  print("=== TIMESTAMP === :", datetime.datetime.now())


def visualize_spectrogram(data_sequence, batch, index):
  x,y = data_sequence.__getitem__(batch)
  print("Current sample: ", ID2LABEL(np.argmax(y[index])))
  plt.pcolormesh(x[index,:,:,0])
  plt.show()

