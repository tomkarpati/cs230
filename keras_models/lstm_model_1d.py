import tensorflow as tf

from keras_models import keras_model

# Implement baseline in Keras
class LstmModel1D(keras_model.KerasModel):
    
  def __init__(self,
               hparams,
               model_dir,
               verbose=False):

    super().__init__(hparams, model_dir, verbose)

    # reshape and squeeze this to be [num_samples, sequence, features]
    def lstm_reshape_layer(x):
      return tf.transpose(x, perm=[0, 2, 1])

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
    x = tf.keras.layers.Lambda(lstm_reshape_layer)(X_in)
    if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)
    
    # Repeat 4 times
    for i in range(self.hparams['num_conv_layers']):
      # Call conv1d (we have powers of 2 increase each iteration)
      # This trains single kernel over all frequencies
      nfilters = 16 * (2 ** i)
      x = tf.keras.layers.Conv1D(filters=nfilters,
                                 kernel_size=3,
                                 strides=1,
                                 use_bias=False,
                                 activation=None,
                                 padding='same',
                                 name='conv1d_h{}'.format(i))(x)
      # Run batch normalization per conv layer before activation
      if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('relu')(x)

    # LSTM layer
    for i in range(self.hparams['num_lstm_hidden_layers']):
      # Call lstm layer multiple times as needed
      x = lstm_layer(x,
                     num_nodes=128,
                     return_sequences=True,
                     name='lstm_h{}'.format(i))
      if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)
      
    x = lstm_layer(x,
                   num_nodes=128,
                   name='lstm')
    if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)
    
    # Get the logits for softmax output
    logits = tf.keras.layers.Dense(self.hparams['num_classes'],
                                   activation='softmax',
                                   use_bias=True,
                                   name='dense_softmax')(x)

    self.model = tf.keras.models.Model(inputs=X_in, outputs=logits)


    self.compile()

