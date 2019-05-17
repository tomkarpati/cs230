import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import signal
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
import tqdm

POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}
WAVE_SAMPLES = 16000 # expected length of example

def data_generator(data,
                   params,
                   mode='train'):
  # TK - This is shimed to use the standard data_generator for the project
  def generator():
    if mode == 'train':
      np.random.shuffle(data)
    for d in data:
      # d is a dictionary of the example
      #try:
        l = len(d['data'])
        w = np.zeros((WAVE_SAMPLES), dtype=d['data'].dtype)
        if l > WAVE_SAMPLES:
          beg = np.random.randint(0, l - WAVE_SAMPLES) # pull random window from data
          w = d['data'][beg:beg+WAVE_SAMPLES]
        else:
          w[0:l] = d['data'] # Fill int he begining with the sample (rest is zero)
        label = d['class']
        if label == '_background_noise_':
          label = 'silence'
        if label not in POSSIBLE_LABELS:
          label = 'unknown'
          
        yield dict(
            target = np.int32(name2id[label]),
            fname = np.string_(d['filename']),
            wav = w
        )
      #except Exception as err:
      #  print ("ERROR: ", err, d['class'], d['filename'])

  return generator

def model(x, params, is_training):
    x = layers.batch_norm(x, is_training=is_training)
    for i in range(4):
        x = layers.conv2d(
            x, 16 * (2 ** i), 3, 1,
            activation_fn=tf.nn.elu,
            normalizer_fn=layers.batch_norm if params.use_batch_norm else None,
            normalizer_params={'is_training': is_training}
        )
        x = layers.max_pool2d(x, 2, 2)

    # just take two kind of pooling and then mix them, why not :)
    mpool = tf.reduce_max(x, axis=[1, 2], keep_dims=True)
    apool = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

    x = 0.5 * (mpool + apool)
    # we can use conv2d 1x1 instead of dense
    x = layers.conv2d(x, 128, 1, 1, activation_fn=tf.nn.elu)
    x = tf.nn.dropout(x, keep_prob=params.keep_prob if is_training else 1.0)
    
    # again conv2d 1x1 instead of dense layer
    logits = layers.conv2d(x, params.num_classes, 1, 1, activation_fn=None)
    return tf.squeeze(logits, [1, 2])

# features is a dict with keys: tensors from our datagenerator
# labels also were in features, but excluded in generator_input_fn by target_key

def model_handler(features, labels, mode, params, config):
    # Im really like to use make_template instead of variable_scopes and re-usage
    extractor = tf.make_template(
        'extractor', model,
        create_scope_now_=True,
    )
    # wav is a waveform signal with shape (16000, )
    wav = features['wav']
    # we want to compute spectograms by means of short time fourier transform:
    specgram = signal.stft(
        wav,
        400,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
        160,  # 16000 * 0.010 -- default stride
    )
    # specgram is a complex tensor, so split it into abs and phase parts:
    phase = tf.angle(specgram) / np.pi
    # log(1 + abs) is a default transformation for energy units
    amp = tf.log1p(tf.abs(specgram))
    
    x = tf.stack([amp, phase], axis=3) # shape is [bs, time, freq_bins, 2]
    x = tf.to_float(x)  # we want to have float32, not float64

    logits = extractor(x, params, mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        # some lr tuner, you could use move interesting functions
        def learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
            learning_rate_decay_fn=learning_rate_decay_fn,
            clip_gradients=params.clip_gradients,
            variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        specs = dict(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        prediction = tf.argmax(logits, axis=-1)
        acc, acc_op = tf.metrics.mean_per_class_accuracy(
            labels, prediction, params.num_classes)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        specs = dict(
            mode=mode,
            loss=loss,
            eval_metric_ops=dict(
                acc=(acc, acc_op),
            )
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'label': tf.argmax(logits, axis=-1),  # for probability just take tf.nn.softmax()
            'target': features['target'],
            'fname': features['fname'],
        }
        specs = dict(
            mode=mode,
            predictions=predictions,
        )
    return tf.estimator.EstimatorSpec(**specs)


def create_model(config=None, hparams=None):
    return tf.estimator.Estimator(
        model_fn=model_handler,
        config=config,
        params=hparams,
    )

def baseline(out_dir="."):
  
  params=dict(
      seed=2018,
      batch_size=64,
      keep_prob=0.5,
      learning_rate=1e-3,
      clip_gradients=15.0,
      use_batch_norm=True,
      num_classes=len(POSSIBLE_LABELS),
  )

  hparams = tf.contrib.training.HParams(**params)
  os.makedirs(os.path.join(out_dir, 'eval'), exist_ok=True)
  model_dir = out_dir

  run_config = tf.contrib.learn.RunConfig(model_dir=model_dir)
  return run_config, hparams

def train(run_config,
          training_data,
          validation_data,
          hparams):
  train_input_fn = generator_input_fn(
      x=data_generator(training_data, hparams, 'train'),
      target_key='target',  # you could leave target_key in features, so labels in model_handler will be empty
      batch_size=hparams.batch_size, shuffle=True, num_epochs=None,
      queue_capacity=3 * hparams.batch_size + 10, num_threads=1,
  )

  val_input_fn = generator_input_fn(
      x=data_generator(validation_data, hparams, 'val'),
      target_key='target',
      batch_size=hparams.batch_size, shuffle=True, num_epochs=None,
      queue_capacity=3 * hparams.batch_size + 10, num_threads=1,
  )
            
  def _create_my_experiment(run_config, hparams):
    exp = tf.contrib.learn.Experiment(
        estimator=create_model(config=run_config, hparams=hparams),
        train_input_fn=train_input_fn,
        eval_input_fn=val_input_fn,
        train_steps=10000, # just randomly selected params
        eval_steps=200,  # read source code for steps-epochs ariphmetics
        train_steps_per_iteration=1000,
    )
    return exp

  tf.contrib.learn.learn_runner.run(
      experiment_fn=_create_my_experiment,
      run_config=run_config,
      schedule="continuous_train_and_eval",
      hparams=hparams)
  
def test(run_config,
         test_data,
         hparams):

  test_input_fn = generator_input_fn(
      x=data_generator(test_data, hparams, 'test'),
      batch_size=hparams.batch_size, 
      shuffle=False, 
      num_epochs=1,
      queue_capacity= 10 * hparams.batch_size, 
      num_threads=1,
  )

  model = create_model(config=run_config, hparams=hparams)
  it = model.predict(input_fn=test_input_fn)

  correct = 0
  total = 0
  for pred_dict in tqdm.tqdm(it):
    fname, target, label  = pred_dict['fname'].decode(), id2name[pred_dict['target']], id2name[pred_dict['label']]
    total += 1
    if target == label: correct += 1
    #else: print(fname, target, label)

  print ("Accuracy: {} [{}/{}]".format((correct/total), correct, total))
