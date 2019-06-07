import datetime

import matplotlib.pyplot as plt

def log_timestamp():
  print("=== TIMESTAMP === :", datetime.datetime.now())


def visualize_waveform(data):
  plt.plot(data)
  plt.xlabel('Time (samples)')
  plt.ylabel('Amplitude')
  plt.title('Audio Waveform')
  plt.show()
  

def visualize_spectrogram(data, label):
  print("Current sample: ", label)
  fig, a = plt.subplots()
  im = a.pcolormesh(data)
  plt.xlabel('Time (window index)')
  plt.ylabel('Frequency')

  fig.colorbar(im, ax=a, orientation='vertical')
  
  plt.title('Audio Spectrogram')
  plt.show()

