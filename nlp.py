import numpy as np
import tensorflow as tf
from transformer import Transformer
import time
import tensorflow_text as text
#tf.config.run_functions_eagerly(True)
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
EPOCHS = 20
steps_per_epoch = 100

_VOCAB = [
    # Special tokens
    b"[UNK]", b"[MASK]", b"[RANDOM]", b"[CLS]", b"[SEP]",
    # Suffixes
    b"##ack", b"##ama", b"##ger", b"##gers", b"##onge", b"##pants",  b"##uare",
    b"##vel", b"##ven", b"an", b"A", b"Bar", b"Hates", b"Mar", b"Ob",
    b"Patrick", b"President", b"Sp", b"Sq", b"bob", b"box", b"has", b"highest",
    b"is", b"office", b"the",
]

examples = {
    "text_a": [
      "Sponge bob Squarepants is an Avenger",
      "Marvel Avengers"
    ],
    "text_b": [
     "Barack Obama is the President.",
     "President is the highest office"
  ],
}

_START_TOKEN = _VOCAB.index(b"[CLS]")
_END_TOKEN = _VOCAB.index(b"[SEP]")
_MASK_TOKEN = _VOCAB.index(b"[MASK]")
_RANDOM_TOKEN = _VOCAB.index(b"[RANDOM]")
_UNK_TOKEN = _VOCAB.index(b"[UNK]")
_MAX_SEQ_LEN = 8
_MAX_PREDICTIONS_PER_BATCH = 5

_VOCAB_SIZE = len(_VOCAB)

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
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

lookup_table = tf.lookup.StaticVocabularyTable(
    tf.lookup.KeyValueTensorInitializer(
      keys=_VOCAB,
      key_dtype=tf.string,
      values=tf.range(
          tf.size(_VOCAB, out_type=tf.int64), dtype=tf.int64),
          value_dtype=tf.int64
        ),
      num_oov_buckets=1
)

bert_tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=30000,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=' ',
    char_level=False,
    oov_token=None,
    document_count=0
)
bert_tokenizer.fit_on_texts([ 
     "Sponge bob Squarepants is an Avenger",
     "Marvel Avengers",
     "Barack Obama is the President.",
     "President is the highest office"])


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
def losss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=2),dtype=tf.int32))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
#train_loss = tf.keras.metrics.Mean(name='train_loss')
#$train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    pe_input = 100,
    pe_target = 100,
    input_vocab_size=30000,
    target_vocab_size=30000,
    rate=dropout_rate)
def make_batches(ds):
  return (
      ds
      .cache()
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
      .filter(filter_max_tokens)
      .prefetch(tf.data.AUTOTUNE))


#train_batches = make_batches()
#val_batches = make_batches(val_examples)

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


#@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  print(tar)
  print()
  tar_inp = [tar[:-1]]
  tar_real = [tar[1:]]
  print(tar)
  print(inp)
  print(tar_inp)
  print(tar_real)
  inp = [inp]
  inp = np.array(inp)
  tar = np.array(tar)
  tar_inp = np.array(tar_inp)
  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],
                                 training = True)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))
'''
for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate([examples['text_a']]):
    print('-----------------------------------')
    print(inp)
    print(bert_tokenizer.texts_to_sequences([inp]))
    #print(bert_tokenizer.texts_to_sequences(inp)[0][0])
    print('++++++++++++++++++++++++++++++++=')
    train_step(bert_tokenizer.texts_to_sequences([inp])[0], bert_tokenizer.texts_to_sequences([tar])[0])


    #if batch % 50 == 0:
    #  print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  #if (epoch + 1) % 5 == 0:
  #  ckpt_save_path = ckpt_manager.save()
  #  print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

  #print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
  

  #print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
'''
temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
#train_loss = tf.keras.metrics.Mean(name='train_loss')

@tf.function
def train_multiple_steps(iterator,iterator_test, steps):
  """The step function for one training step."""
  #train_loss = tf.keras.metrics.Mean(name='train_loss')
  #train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
  def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
  def step_fn(inputs):
      """The computation to run on each TPU device."""
      #train_loss = tf.keras.metrics.Mean(name='train_loss')
      #train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

      #print('------------------')
      #print(inputs)
      #print('----------------------')
      inp,tar = inputs
      #print(tar)
      #print()
      tar_inp = [tar[:-1]]
      tar_real = [tar[1:]]
      #print(tar)
      #print(inp)
      #print(tar_inp)
      #print(tar_real)
      #print(tf.convert_to_tensor(tar_inp,dtype=tf.int32))
      inp = [inp]
      #inp = tf.Variable(inp,dtype=tf.int32)
      #tar = tf.Variable(tar,dtype=tf.int32)
      #tar_inp = tf.Variiable(tar_inp,dtype=tf.int32)
    
      with tf.GradientTape() as tape:
         predictions, _ = transformer([tf.convert_to_tensor(inp), tf.convert_to_tensor(tar_inp,dtype=tf.int32)],
                                 training = True)
         loss = loss_function(tar_real, predictions)
         #tf.print(loss)  
      gradients = tape.gradient(loss, transformer.trainable_variables)
      optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
      return predictions
      #train_loss(loss)
      #train_accuracy(accuracy_function(tf.cast(tar_real,dtype=tf.int32), tf.cast(predictions,dtype=tf.int32)))
  print(iterator)
  print('++++++++++++++')
  iterator = iter([iterator])

  for _ in tf.range(steps):
   #print('===========================')
   strategy.run(step_fn, args=(next(iterator),))
  def test(iterator_test):
     inp = iterator_test[0]
     tar = iterator_test[1]
     #print(tar)
     #print()
     tar_inp = [tar[:-1]]
     tar_real = [tar[1:]]
     inp = [inp]
  
     bert_tokenizer.sequences_to_texts(transformer([tf.convert_to_tensor(inp), tf.convert_to_tensor(tar_inp,dtype=tf.int32)],
                                 training = True))
  #test(iterator_test)
  #print('+=++++++=====++=+')
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
#per_replica_batch_size = batch_size // strategy.num_replicas_in_sync

#train_dataset = strategy.experimental_distribute_datasets_from_function(
#    lambda _: get_dataset(per_replica_batch_size, is_training=True))
#train_iterator = iter(train_dataset)
for (batch, (inp, tar)) in enumerate([examples['text_a']]):

 train_multiple_steps([bert_tokenizer.texts_to_sequences([inp])[0], bert_tokenizer.texts_to_sequences([tar])[0]],[bert_tokenizer.texts_to_sequences([inp])[0], bert_tokenizer.texts_to_sequences([tar])[0]], tf.convert_to_tensor(steps_per_epoch))

def test(iterator_test):
     inp = iterator_test[0]
     tar = iterator_test[1]
     #print(tar)
     #print()
     tar_inp = [tar[:-1]]
     tar_real = [tar[1:]]
     inp = [inp]
     print('==============')
     #print(transformer([tf.convert_to_tensor(inp), tf.convert_to_tensor(tar_inp,dtype=tf.int32)],
     #                            training = True)[0].numpy().tolist()[0][0])
     transformer_output = transformer([tf.convert_to_tensor(inp), tf.convert_to_tensor(tar_inp,dtype=tf.int32)],
                                 training = True)[0].numpy().tolist()[0]
     for i in range(len(transformer_output[0])):
         transformer_output[0][i] = int((transformer_output[0][i]+1.0)*30000)
     #print(transformer_output)
     #print(bert_tokenizer.sequences_to_texts(transformer_output))
     print('+++++++++++++++++++++++==')
     print(tf.convert_to_tensor(inp).numpy().tolist())
test([bert_tokenizer.texts_to_sequences([inp])[0], bert_tokenizer.texts_to_sequences([tar])[0]])

#fn_out, _ = sample_transformer([temp_input, temp_target], training=False)

#print(fn_out.shape)
