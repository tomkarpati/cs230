import os
import datetime

import tensorflow as tf

import json

# Override the tensorboard callback to dump out our learning rate
class LRTensorBoard(tf.keras.callbacks.TensorBoard):
  def __init__(self, log_dir):  # add other arguments to __init__ if you need
    super().__init__(log_dir=log_dir)

  def on_epoch_end(self, epoch, logs=None):
    logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
    super().on_epoch_end(epoch, logs)
        

# Create a base class for Keras models
class KerasModel:
    
  def __init__(self,
               hparams,
               model_dir,
               verbose=False):
    """Create the Keras Model"""

    self.hparams = hparams
    self.verbose = verbose
    self.model_dir = model_dir

    # Generate tracing data to use to profile
    self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    self.run_metadata= tf.RunMetadata()

    # Setup callbacks
    self.callbacks = []

    # define a callback for checkpointing the model
    path = self.model_dir+'/checkpoints'
    os.makedirs(path, exist_ok=True)
    path = path+'/model-{epoch:03d}.hdf5'
    self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                             verbose=self.verbose))

    # define callbacks for tensorboard
    path = self.model_dir+'/tensorboard'
    os.makedirs(path, exist_ok=True)
    self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=path,
                                                         histogram_freq=100,
                                                         write_images=True,
                                                         update_freq=100*self.hparams['batch_size']))
    path = self.model_dir+'/tensorboard_lr'
    os.makedirs(path, exist_ok=True)
    self.callbacks.append(LRTensorBoard(log_dir=path))
    
    # Write out results per epoch
    path = self.model_dir+'/csv_results'
    os.makedirs(path, exist_ok=True)
    path = path+'/results.csv'
    self.callbacks.append(tf.keras.callbacks.CSVLogger(filename=path, separator=',', append=True))

    #def learning_rate_decay (x) :
    #  return self.hparams * (0.9 ** self.hparams['lr_decay'])
    
    #self.callbacks.append(tf.keras.callbacks.LearningRateScheduler(learning_rate_decay))
    
    print (self.callbacks)

    if self.hparams['optimizer'] == 'adam':
      self.optimizer = tf.keras.optimizers.Adam(lr=self.hparams['lr'],
                                                decay=self.hparams['lr_decay'])

    
  def compile(self):
    self.model.compile(loss=self.hparams['loss'],
                       optimizer=self.optimizer,
                       metrics=['categorical_accuracy'],
                       options=self.run_options,
                       run_metadata=self.run_metadata)
    self.dump_info()


  def train(self,
            training_sequence,
            validation_sequence,
            verbose=False):
      
    return self.model.fit_generator(generator=training_sequence,
                                    steps_per_epoch=None,
                                    epochs=self.hparams['epochs'],
                                    verbose=2,
                                    workers=self.hparams['threads'],
                                    use_multiprocessing=self.hparams['multiprocess'],
                                    callbacks=self.callbacks,
                                    validation_data=validation_sequence)

  def test(self,
            test_sequence,
            verbose=False):
      
    return self.model.evaluate_generator(generator=test_sequence,
                                         steps=None,
                                         verbose=2,
                                         workers=self.hparams['threads'],
                                         use_multiprocessing=self.hparams['multiprocess'])

  def dump_info(self):
    # Dump hyperparameter information
    print("Hyperparameters: ")
    print(self.hparams)
    d = self.model_dir+'/parameters'
    os.makedirs(d, exist_ok=True)
    path = d+'/hyperparameters.json'
    f = open(path,"w")
    json.dump(self.hparams, f)
    f.close()

    # Dump model information
    self.model.summary()
    path = d+'/model.txt'
    f = open(path,"w")
    self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.close()
      

  def dump_profiling_data(self):
    from tensorflow.python.client import timeline
    tl = timeline.Timeline(self.run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    path = self.model_dir+'/profiling'
    os.makedirs(path, exist_ok=True)
    with open(path+'/timeline.json', 'w') as f:
        f.write(ctf)
  

