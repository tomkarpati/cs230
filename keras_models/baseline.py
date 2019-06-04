import tensorflow as tf

from keras_models import keras_model

# Implement baseline in Keras
class BaselineModel(keras_model.KerasModel):
    
  def __init__(self,
               hparams,
               model_dir,
               verbose=False):

    super().__init__(hparams, model_dir, verbose)
    
    X_in = tf.keras.layers.Input(shape=self.hparams['input_shape'], name='input')
    x = tf.keras.layers.Lambda(lambda x : tf.expand_dims(x, axis=3))(X_in)
    if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)
    
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
      if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('elu')(x)
      x = tf.keras.layers.MaxPooling2D(pool_size=2,
                                       strides=2,
                                       padding='same',
                                       name='max_pool_h{}'.format(i))(x)

    # Flatten out the convolution layers
    x = tf.keras.layers.Flatten()(x)
    # Dense layer
    x = tf.keras.layers.Dense(128,
                              activation='elu',
                              use_bias=True,
                              name='dense')(x)
    # perform dropout
    x = tf.keras.layers.Dropout(rate=(1-self.hparams['keep_prob']))(x)
    
    # Get the logits for softmax output
    logits = tf.keras.layers.Dense(self.hparams['num_classes'],
                                   activation='softmax',
                                   use_bias=True,
                                   name='dense_softmax')(x)

    self.model = tf.keras.models.Model(inputs=X_in, outputs=logits)


    self.compile()

