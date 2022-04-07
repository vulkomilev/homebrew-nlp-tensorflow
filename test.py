import tensorflow as tf

import os
import tensorflow_datasets as tfds
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='homebrewnlp-zerox')
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

with tf.device('/TPU:0'):
  c = tf.matmul(a, b)

print("c device: ", c.device)
print(c)
strategy = tf.distribute.TPUStrategy(resolver)
@tf.function
def matmul_fn(x, y):
  z = tf.matmul(x, y)
  return z

z = strategy.run(matmul_fn, args=(a, b))
print(z)
def create_model():
  return tf.keras.Sequential(
      [tf.keras.layers.Conv2D(256, 3, activation='relu', input_shape=(28, 28, 1)),
       tf.keras.layers.Conv2D(256, 3, activation='relu'),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(256, activation='relu'),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10)])
def get_dataset(batch_size, is_training=True):
  split = 'train' if is_training else 'test'
  dataset, info = tfds.load(name='mnist', split=split, with_info=True,
                            as_supervised=True, try_gcs=True)

  # Normalize the input data.
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label

  dataset = dataset.map(scale)

  # Only shuffle and repeat the dataset in training. The advantage of having an
  # infinite dataset for training is to avoid the potential last partial batch
  # in each epoch, so that you don't need to think about scaling the gradients
  # based on the actual batch size.
  if is_training:
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat()

  dataset = dataset.batch(batch_size)
  #print('dataset',dataset)
  return dataset
  with strategy.scope():
    model = create_model()
  
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

batch_size = 200
steps_per_epoch = 60000 // batch_size
validation_steps = 10000 // batch_size

train_dataset = get_dataset(batch_size, is_training=True)
test_dataset = get_dataset(batch_size, is_training=False)


with strategy.scope():
  model = create_model()
  model.compile(optimizer='adam',
                # Anything between 2 and `steps_per_epoch` could help here.
                steps_per_execution = 50,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
model.fit(train_dataset,
          epochs=5,
          steps_per_epoch=steps_per_epoch,
          validation_data=test_dataset,
          validation_steps=validation_steps)

# Create the model, optimizer and metrics inside the strategy scope, so that the
# variables can be mirrored on each device.
with strategy.scope():
  model = create_model()
  steps_per_eval = 10000 // batch_size
  optimizer = tf.keras.optimizers.Adam()
  training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
  training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      'training_accuracy', dtype=tf.float32)
  print('train_dataset',train_dataset)
  #train_iterator = iter(train_dataset)


#for epoch in range(5):
#  print('Epoch: {}/5'.format(epoch))
#
#  for step in range(steps_per_epoch):
#    train_step(train_iterator)
#  print('Current step: {}, training loss: {}, accuracy: {}%'.format(
#      optimizer.iterations.numpy(),
#      round(float(training_loss.result()), 4),
#      round(float(training_accuracy.result()) * 100, 2)))
#  training_loss.reset_states()
#  training_accuracy.reset_states()

#optimizer = tf.keras.optimizers.Adam()
#training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
#training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
#      'training_accuracy', dtype=tf.float32)
@tf.function
def train_multiple_steps(iterator, steps):
  """The step function for one training step."""

  def step_fn(inputs):
    """The computation to run on each TPU device."""
    images, labels = inputs
    with tf.GradientTape() as tape:
      logits = model(images, training=True)
      loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, logits, from_logits=True)
      loss = tf.nn.compute_average_loss(loss, global_batch_size=batch_size)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    training_loss.update_state(loss * strategy.num_replicas_in_sync)
    training_accuracy.update_state(labels, logits)
  iterator = iter(iterator)

  for _ in tf.range(steps):
    strategy.run(step_fn, args=(next(iterator),))

# Convert `steps_per_epoch` to `tf.Tensor` so the `tf.function` won't get 
# retraced if the value changes.
#train_multiple_steps(train_iterator, tf.convert_to_tensor(steps_per_epoch))


#print('Current step: {}, training loss: {}, accuracy: {}%'.format(
#      optimizer.iterations.numpy(),
#      round(float(training_loss.result()), 4),
#      round(float(training_accuracy.result()) * 100, 2)))
#exit(0)
#optimizer = tf.keras.optimizers.Adam()
#training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
#training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
#      'training_accuracy', dtype=tf.float32)

# Calculate per replica batch size, and distribute the datasets on each TPU
# worker.
per_replica_batch_size = batch_size // strategy.num_replicas_in_sync

#train_dataset = strategy.experimental_distribute_datasets_from_function(
#    lambda _: get_dataset(per_replica_batch_size, is_training=True))

@tf.function
def train_step(iterator):
  """The step function for one training step."""

  def step_fn(inputs):
    """The computation to run on each TPU device."""
    images, labels = inputs
    with tf.GradientTape() as tape:
      logits = model(images, training=True)
      loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, logits, from_logits=True)
      loss = tf.nn.compute_average_loss(loss, global_batch_size=batch_size)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    training_loss.update_state(loss * strategy.num_replicas_in_sync)
    training_accuracy.update_state(labels, logits)
  iterator = iter(iterator)
  strategy.run(step_fn, args=(next(iterator),))
#train_iterator = iter(train_dataset)
train_multiple_steps(train_dataset, tf.convert_to_tensor(steps_per_epoch))

#print('Current step: {}, training loss: {}, accuracy: {}%'.format(
#      optimizer.iterations.numpy(),
#      round(float(training_loss.result()), 4),
#      round(float(training_accuracy.result()) * 100, 2)))

