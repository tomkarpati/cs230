import tensorflow as tf

from keras_models import keras_model

# Implement baseline in Keras
class LstmModel(keras_model.KerasModel):
    
  def __init__(self,
               hparams,
               model_dir,
               verbose=False):

    super().__init__(hparams, model_dir, verbose)

    # Define the standard vs CuDNNLSTM dependent on GPU availability
    def lstm_layer(x,
                   num_nodes,
                   return_sequences=False,
                   name='lstm'):
      if self.hparams['has_gpu']:
        return tf.keras.layers.CuDNNLSTM(num_nodes,
                                         return_sequences=return_sequences,
                                         name=name)(x)
      else:
        return tf.keras.layers.LSTM(num_nodes,
                                    return_sequences=return_sequences,
                                    name=name)(x)
    
    X_in = tf.keras.layers.Input(shape=self.hparams['input_shape'], name='input')
    x = tf.keras.layers.Lambda(lambda x : tf.expand_dims(x, axis=2))(X_in)
    if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)

    # Perform initial 1d convolution to learn basis functions for spectrogram
    x = tf.keras.layers.Conv1D(filters=self.hparams['basis_filters'],
                               kernel_size=self.hparams['basis_kernel_size'],
                               strides=self.hparams['basis_kernel_stride'],
                               use_bias=False,
                               activation=None,
                               padding='same',
                               name='basis_conv1d')(x)
    # Run batch normalization per conv layer before activation
    if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if self.hparams['dropout']: x = tf.keras.layers.Dropout(rate=1-self.hparams['keep_prob'],
                                                            seed=self.hparams['seed'])(x)
    
    # Repeat n times
    for i in range(self.hparams['num_conv_layers']):
      # Call conv1d (we have powers of 2 increase each iteration)
      # This trains single kernel over all frequencies
      nfilters = 16 * (2 ** i)
      x = tf.keras.layers.Conv1D(filters=nfilters,
                                 kernel_size=self.hparams['kernel_size'][i],
                                 strides=self.hparams['kernel_stride'][i],
                                 use_bias=False,
                                 activation=None,
                                 padding='same',
                                 name='conv1d_h{}'.format(i))(x)
      # Run batch normalization per conv layer before activation
      if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('relu')(x)
      if self.hparams['dropout']: x = tf.keras.layers.Dropout(rate=1-self.hparams['keep_prob'],
                                                              seed=self.hparams['seed'])(x)

    # LSTM layer
    for i in range(self.hparams['num_lstm_hidden_layers']):
      # Call lstm layer multiple times as needed
      x = lstm_layer(x,
                     num_nodes=self.hparams['lstm_features'],
                     return_sequences=True,
                     name='lstm_h{}'.format(i))
      if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)
      if self.hparams['dropout']: x = tf.keras.layers.Dropout(rate=1-self.hparams['keep_prob'],
                                                              seed=self.hparams['seed'])(x)
      
    x = lstm_layer(x,
                   num_nodes=self.hparams['lstm_features'],
                   name='lstm')
    if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)
    if self.hparams['dropout']: x = tf.keras.layers.Dropout(rate=1-self.hparams['keep_prob'],
                                                            seed=self.hparams['seed'])(x)
    
    # Get the logits for softmax output
    logits = tf.keras.layers.Dense(self.hparams['num_classes'],
                                   activation='softmax',
                                   use_bias=True,
                                   name='dense_softmax')(x)

    self.model = tf.keras.models.Model(inputs=X_in, outputs=logits)


    self.compile()

