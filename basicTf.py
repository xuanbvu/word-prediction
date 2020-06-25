import tensorflow as tf

import numpy as np
import os
import time

# shakespeare file
# path_to_file = tf.keras.utils.get_file('fb.txt', './fb.txt')
path_to_file = os.path.abspath('clark.txt')

'''PROCESS TEXT'''
# read file
text = open(path_to_file, "rb").read().decode(encoding="utf-8")
# unique characters
vocab = sorted(set(text))

# maps characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# represents text as numbers
# First -> 18 47 56 57 58
text_as_int = np.array([char2idx[c] for c in text])

'''CREATE TRAINING EXAMPLES AND TARGETS'''
# if text is "Hello", input is "Hell" and target is "ello"
# target is input shifted one character to the right so (seq_length+1)
seq_length = 100
examples_per_epoch = len(text) // (seq_length+1)

# convert the text into character indices
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# convert characters into sequences
# basically splits the text into sequences of seq_length chars
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# form input and target from sequence


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


# store as tuple of (input, target) ...i think
dataset = sequences.map(split_input_target)

'''CREATE TRAINING BATCHES'''
# before using this data, need to shuffle and pack it into batches
# tf doesn't shuffle entire sequence in memory, it maintains a buffer
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

'''BUILD MODEL'''
# three layers are used to define the model:
# input layer (trainable lookup table mapping numbers of each character to a vector) with embedding_dim dimensions
# RNN with size of rnn_units
# output layer with vocab_size outputs
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True,
                            stateful=True, recurrent_initializer="glorot_uniform"),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)


'''TRAIN MODEL'''


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer="adam", loss=loss)

# directory where checkpoints will be saved
checkpoint_dir = "./training_checkpoints"
# name of checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, save_weights_only=True)

EPOCHS = 30
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

'''GENERATE TEXT'''
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


def generate_text(model, start_string):
    # number of characters to generate
    num_generate = 140

    # converting start_string to numbers
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # empty string to store results
    text_generated = []

    # low temp = more predictable; high temp = more surprising
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        # predict character returned by model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        # pass predicted character as the next input to model along w the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return (start_string + "".join(text_generated))


print(generate_text(model, start_string=u"a"))
