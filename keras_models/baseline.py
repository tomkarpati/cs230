import os

import tensorflow as tf
import tensorflow.keras.models
import tensorflow.keras.callbacks

# Implement baseline in Keras
class BaselineModel:
    
  def __init__(self,
               hparams,
               model_dir,
               verbose=False):
    """Create the Keras Model"""

    self.hparams = hparams
    self.verbose = verbose
    self.model_dir = model_dir
    
    X_in = tf.keras.layers.Input(shape=self.hparams['input_shape'], name='input')
    x = tf.keras.layers.BatchNormalization()(X_in)
    # Repeat 4 times
    for i in range(4):
      # Call conv2d (we have powers of 2 increase each iteration)
      nfilters = 16 * (2 ** i)
      x = tf.keras.layers.Conv2D(filters=nfilters,
                                 kernel_size=3,
                                 strides=1,
                                 use_bias=False,
                                 activation=None,
                                 padding='same',
                                 name='conv2d_h{}'.format(i))(x)
      # Run batch normalization per conv layer before activation
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('elu')(x)
      x = tf.keras.layers.MaxPooling2D(pool_size=2,
                                       strides=2,
                                       padding='same',
                                       name='max_pool_h{}'.format(i))(x)

    # Flatten out the convolution layers
    x = tf.keras.layers.Flatten()(x)
    # Dense layer
    x = tf.keras.layers.Dense(128,
                              name='dense')(x)
    # perform dropout
    x = tf.keras.layers.Dropout(rate=(1-self.hparams['keep_prob']))(x)
    
    # Get the logits for softmax output
    logits = tf.keras.layers.Dense(self.hparams['num_classes'],
                                   name='dense_softmax')(x)

    self.model = tf.keras.models.Model(inputs=X_in, outputs=logits)

    # Generate tracing data to use to profile
    self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    self.run_metadata= tf.RunMetadata()
    
    self.model.compile(loss=self.hparams['loss'],
                       optimizer=self.hparams['optimizer'],
                       metrics=['categorical_accuracy'],
                       options=self.run_options,
                       run_metadata=self.run_metadata)
    self.model.summary()
    
    # Setup callbacks
    self.callbacks = []

    # define a callback for checkpointing the model
    path = self.model_dir+'/checkpoints'
    os.makedirs(path, exist_ok=True)
    path = path+'/model-{epoch:03d}.hdf5'
    self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                             verbose=verbose))
    path = self.model_dir+'/tensorboard'
    os.makedirs(path, exist_ok=True)
    self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=path,
                                                         histogram_freq=100,
                                                         update_freq=100))
    print (self.callbacks)

  def train(self,
            training_sequence,
            validation_sequence,
            verbose=False):
      
    return self.model.fit_generator(generator=training_sequence,
                                    steps_per_epoch=None,
                                    epochs=self.hparams['epochs'],
                                    verbose=2,
                                    #workers=10,
                                    #use_multiprocessing=True,
                                    callbacks=self.callbacks,
                                    validation_data=validation_sequence)

  def dump_profiling_data(self):
    from tensorflow.python.client import timeline
    tl = timeline.Timeline(self.run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    path = self.model_dir+'/profiling'
    os.makedirs(path, exist_ok=True)
    with open(path+'/timeline.json', 'w') as f:
        f.write(ctf)
  

