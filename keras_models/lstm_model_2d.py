import tensorflow as tf

from keras_models import keras_model

# Implement baseline in Keras
class LstmModel(keras_model.KerasModel):
    
  def __init__(self,
               hparams,
               model_dir,
               verbose=False):

    super().__init__(hparams, model_dir, verbose)
    
    # reshape this to be [num_samples, sequence, features]
    def lstm_transpose_layer(x):
      return tf.transpose(x, perm=[0, 2, 1])

    # Flatten out all of the features and filter banks together
    def lstm_reshape_layer(x):
      shape = tf.shape(x)
      return tf.keras.layers.Reshape(target_shape=[65, -1])(x)

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
    x = tf.keras.layers.Lambda(lstm_transpose_layer)(X_in)
    x = tf.keras.layers.Lambda(lambda x : tf.expand_dims(x, axis=3))(x)
    if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)
    
    # Repeat n times
    for i in range(self.hparams['num_conv_layers']):
      # Call conv2d (we have powers of 2 increase each iteration)
      nfilters = 16 * (2 ** i)
      x = tf.keras.layers.Conv2D(filters=nfilters,
                                 kernel_size=self.hparams['kernel_size'][i],
                                 strides=self.hparams['kernel_stride'][i],
                                 use_bias=False,
                                 activation=None,
                                 padding='same',
                                 name='conv2d_h{}'.format(i))(x)
      # Run batch normalization per conv layer before activation
      if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)
      if self.hparams['dropout']: x = tf.keras.layers.Dropout(rate=1-self.hparams['keep_prob'],
                                                              seed=self.hparams['seed'])(x)
      x = tf.keras.layers.Activation('relu')(x)
      x = tf.keras.layers.MaxPooling2D(pool_size=2,
                                       strides=2,
                                       padding='same',
                                       name='max_pool_h{}'.format(i))(x)

    # Merge all the frequencies and filters together
    print (x.shape.as_list())
    dims = x.shape.as_list()
    print(dims)
    x = tf.keras.layers.Reshape((dims[1], dims[2]*dims[3]))(x)
    print(x.shape.as_list())

    # LSTM layer
    for i in range(self.hparams['num_lstm_hidden_layers']):
      # Call lstm layer multiple times as needed
      x = lstm_layer(x,
                     num_nodes=128,
                     return_sequences=True,
                     name='lstm_h{}'.format(i))
      if self.hparams['batch_norm']: x = tf.keras.layers.BatchNormalization()(x)
      if self.hparams['dropout']: x = tf.keras.layers.Dropout(rate=1-self.hparams['keep_prob'],
                                                              seed=self.hparams['seed'])(x)
      
    x = lstm_layer(x,
                   num_nodes=128,
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

