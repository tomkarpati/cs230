
import re
import os
import glob
import copy

import numpy as np
from scipy.io import wavfile

def collect_stats(dataset):
  size = len(dataset)
  print("Total samples: ", size)
  d = {}
  for example in dataset:
    if example['class'] in d.keys():
      d[example['class']] += 1
    else:
      d[example['class']] = 1
  print("Breakdown of set (percentages of total samples):")
  for k,v in d.items():
    print("  {}: {}".format(k, v*100.0/size))
      
def split_files(data_dir,
                verbose=False):
  """This splits the files into appropriate data sets for training, 
  validation, and test sets.
  The data directory looks like:
    <command>/<speaker_id>_nohash_<utternce_number>.wav"""

  def convert_filelist_to_list(filelist,
                               pattern,
                               data_dir,
                               verbose=False):
    """Return a list of tuples:
    [(label, filename), (...), ...]
    """

    l = []
    for f in filelist:
      ff = f.rstrip()
      r_dir = re.match(pattern['dir'], ff)
      fn = data_dir+'/'+ff
      l.append((r_dir.group(2), fn))

    return l

  # Process data_dir to strip out trailing slash
  if data_dir[-1] == '/': data_dir = data_dir[0:-1]

  pattern = {}
  pattern['dir'] = re.compile("(([^/]*)\/)*?([^/]+\.wav)")
  pattern['file'] = re.compile("([^_]+)_nohash_(\d+)\.wav")
  # Pull all of the files from data_dir
  filelist = glob.glob(data_dir+'/*/*.wav')

  print ("There are", len(filelist), "files in the dataset (", data_dir, ").")
  
  # validation and test sets are in appropriately named
  # files at the top level
  # Anything not in these is in the training set
  fp = open(data_dir+"/validation_list.txt", "r")
  print ("Reading validation samples from:", fp.name)
  validation_filelist = fp.readlines()
  validation_set = convert_filelist_to_list(validation_filelist,
                                            pattern,
                                            data_dir,
                                            verbose)
  fp.close()
  fp = open(data_dir+"/testing_list.txt", "r")
  print ("Reading test samples from:", fp.name)
  test_filelist = fp.readlines()
  test_set = convert_filelist_to_list(test_filelist,
                                      pattern,
                                      data_dir,
                                      verbose)
  fp.close()

  # generate lists of training and test files
  non_training = []
  for t in validation_set:
    non_training.append(t[1])
  for t in test_set:
    non_training.append(t[1])
  
  # Figure out what's in the training set
  training_set = []
  for f in filelist:
    r_dir = re.match(pattern['dir'], f)
    t = (r_dir.group(2), f)
    if f not in non_training:
      if verbose: print("Adding {} to training".format(f))
      training_set.append(t)
    else:
      if verbose: print("Skipping ", f) 
      
  print("This is our data splits:")
  print("  Training set: ", len(training_set))
  print("  Validation set: ", len(validation_set))
  print("  Test set: ", len(test_set))

  return (training_set, validation_set, test_set)

def generate_data(dataset_list):
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

  examples = []
  for k in dataset_list:
    d = {}
    d['filename'] = k[1]
    d['class'] = k[0]
    d['data'] = read_data_from_wav_file(d['filename'])
    examples.append(d)
    
  collect_stats(examples)
  
  return examples

def write_dataset(dataset,
                  name="data",
                  out_dir="."):
  fn = out_dir+"/"+name+".npy"
  print("Writing dataset to {}...".format(fn), end='', flush=True)
  os.makedirs(out_dir, exist_ok=True)
  np.save(out_dir+"/"+name+".npy", dataset)
  print("Done.")

def read_dataset(name="data",
                 in_dir="."):
  fn = in_dir+"/"+name+".npy"
  print("Reading dataset from {}...".format(fn), end='', flush=True)
  assert(os.path.exists(fn))
  dataset = np.load(fn, allow_pickle=True)
  print("Done.")
  return dataset
    
if (__name__ == '__main__'):
  import sys
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
  parser.add_argument("-data_dir", help="Input data location", default="./data")
  parser.add_argument("-out_dir", help="Output data file location", default="./data")
  args = parser.parse_args()
  
  (training_files, validation_files, test_files) = split_files(args.data_dir, args.verbose)
  print("Generating validation data set...")
  validation_data = generate_data(validation_files)
  write_dataset(validation_data, "validation", args.out_dir)
  print("Done.")
  print("Generating test data set...")
  test_data = generate_data(test_files)
  write_dataset(test_data, "test", args.out_dir)
  print("Done.")
  print("Generating training data set...")
  training_data = generate_data(training_files)
  write_dataset(training_data, "training", args.out_dir)
  print("Done.")
