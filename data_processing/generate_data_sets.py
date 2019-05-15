import numpy as np
import re
import os
import glob

from scipy.io import wavfile

def split_files(data_dir,
                verbose=False):
  """This splits the files into appropriate data sets for training, 
  validation, and test sets.
  The data directory looks like:
    <command>/<speaker_id>_nohash_<utternce_number>.wav"""

  def convert_filelist_to_dict(filelist,
                               pattern,
                               data_dir,
                               verbose=False):
    d = {}
    for f in filelist:
      ff = f.rstrip()
      r_dir = re.match(pattern['dir'], ff)
      r_file = re.match(pattern['file'], r_dir.group(3))
      d[(r_dir.group(2), r_file.group(1), r_file.group(2))] = data_dir + '/' + ff

    return d

  def collect_stats(dataset):
    size = len(dataset)
    d = {}
    # Generate a dictionary with all commands
    for k in dataset.keys():
      if k[0] in d.keys(): d[k[0]] += 1
      else: d[k[0]] = 1

    print("Breakdown of set:")
    for k in d.keys():
      d[k] /= size
      print("  ", k, ": ", d[k])
  
  pattern = {}
  pattern['dir'] = re.compile("(([^/]*)\/)*?([^/]+\.wav)")
  pattern['file'] = re.compile("([^_]+)_nohash_(\d+)\.wav")
  # Pull all of the files from data_dir
  filelist = glob.glob(data_dir+'/*/*.wav')

  print ("There are", len(filelist), "files in the dataset (", data_dir, ").")
  
  # validation and test sets are in appropriately named
  # files at the top level
  # Anything not in these is in the training set
  fp = open(data_dir+"validation_list.txt", "r")
  print ("Reading validation samples from:", fp.name)
  validation_filelist = fp.readlines()
  validation_set = convert_filelist_to_dict(validation_filelist,
                                            pattern,
                                            data_dir,
                                            verbose)
  fp.close()
  fp = open(data_dir+"testing_list.txt", "r")
  print ("Reading test samples from:", fp.name)
  test_filelist = fp.readlines()
  test_set = convert_filelist_to_dict(test_filelist,
                                      pattern,
                                      data_dir,
                                      verbose)
  fp.close()

  # Figure out what's in the training set
  training_set = {}
  for f in filelist:
    r_dir = re.match(pattern['dir'], f)
    if (r_dir.group(2) == "_background_noise_"):
      r_file = re.match("(.+)\.wav", r_dir.group(3))
      t = (r_dir.group(2), r_file.group(1))
      if verbose: print (t)
      training_set[t] = f
    else:
      r_file = re.match(pattern['file'], r_dir.group(3))
      t = (r_dir.group(2), r_file.group(1), r_file.group(2))
      if ((t not in validation_set) and
          (t not in test_set)):
        if verbose: print (t)
        training_set[t] = f
      
      
  print("This is our data splits:")
  print("  Training set: ", len(training_set))
  print("  Validation set: ", len(validation_set))
  print("  Test set: ", len(test_set))

  print("Training set statistics:")
  collect_stats(training_set)
  print("Validation set statistics:")
  collect_stats(validation_set)
  print("Test set statistics:")
  collect_stats(test_set)

  return (training_set, validation_set, test_set)

def generate_data(dataset_struct):
  """Convert the dataset structure to a data.
  Returns a dictionary that looks like:
  dict['filename'] filename string
  dict['class'] class string
  dict['data'] np array of size (length,)
  """

  def read_data_from_wav_file(filename,
                              verbose=False):
    """Read a WAV file and return the data.
    May perform any necessary processing.
    """
  
    d = {}
    # Read the file
    rate, data = wavfile.read(filename)
    dmin = min(data)
    dmax = max(data)
    if verbose:
      print("Samples for {}: {} [{},{}]".format(filename,
                                                np.shape(data),
                                                dmin, dmax))
    assert(rate == 16000)
    # Scale this to -1.0,+1.0
    data = data.astype(np.float32) / np.iinfo(np.int16).max
    return data

  examples = {}
  for k in dataset_struct.keys():
    d = {}
    d['filename'] = dataset_struct[k]
    d['class'] = k[0]
    d['data'] = read_data_from_wav_file(d['filename'], verbose=True)
    print(d)
    
  
if (__name__ == '__main__'):
  import sys
  (training_files, validation_files, test_files) = split_files(sys.argv[1])
  training_data = generate_data(training_files)
  validataion_data = generate_data(validation_files)
  test_data = generate_data(test_data)