import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Embedding


# define the Baybayin characters and their English translations
baybayin_characters = ['ᜊ', 'ᜑ', 'ᜏ', 'ᜌ']
latin_script_translations = ['ba', 'ha', 'wa', 'ya']

# create dictionaries to map characters to indices and vice versa
  # in many machine learning tasks, sequences are often represented as numerical indices
  # rather than the actual characters or words
char_to_index = {char: i for i, char in enumerate(baybayin_characters)}
latscr_to_index = {eng: i for i, eng in enumerate(latin_script_translations)}

# dictionary reference for validation test
index_to_latscr = {i: char for i, char in enumerate(latin_script_translations)}

# get vocabulary size
vocab_size = len(baybayin_characters)

# convert characters and translations to numerical representation
X = [[char_to_index[char] for char in word] for word in baybayin_characters]
y = [[latscr_to_index[word]] for word in latin_script_translations]

# determine the maximum length among all input sequences (Baybayin characters)
max_input_length = max(len(word) for word in baybayin_characters)

# determine the maximum length among all output sequences (English translations)
max_output_length = max(len(word) for word in latin_script_translations)

# pad sequences to ensure uniform length
  # padding sequences is necessary to allow sequences to be processed in parallel and by batches
  # neural network models also expect inputs to have a fixed shape so padding is necessary for compatibility
  # variability in sequence lengths is a natural aspect of real-world data
  # not really required in this case pero it's good practice to pad the sequences
X_padded = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_input_length, padding='post') # for inputs
y_padded = np.array(y)

# define the GRU RNN model architecture
model = Sequential([
    # layer responsible for mapping input sequences from indices to embedding vectors
    Embedding(input_dim=vocab_size, output_dim=10, input_length=max_input_length),
    # layer for the gru rnn architecture - indicate dimensionality of output space
      # units - hypermeter. 32 is a common practive for relatively simple tasks/smaller datasets
    GRU(units=32),
    # output layer of model
      # softmax - activation function to produce a probability distribution over the vocabulary
    Dense(units=vocab_size, activation='softmax')
])

# compile the model
  # adam - optimization algorithm for training - adaptive learning rate
  # sparse_categorical_crossentropy - suitable for multi-class classification problems where target labels are integers
      # each baybayin character can be considered as a distinct class kasi, and each instace (character input) belongs to one class label (translation)
  # accuracy - evaluation metric
      # i think enough na to? since single-class translation lang naman ang nature ng problem
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
  # epochs - iterations over the entire dataset
  # verbose - verbosity ng output updates during training
model.fit(X_padded, y_padded, epochs=100, verbose=2)

# save the trained model
model.save('baybayin_to_english_translation_model.h5')

# test the model
test = ['ᜊ', 'ᜑ', 'ᜏ', 'ᜌ']
for input in test:
  test_sequence = np.array([[char_to_index[input]]])
  predicted_probabilities = model.predict(test_sequence)
  predicted_class_index = np.argmax(predicted_probabilities)
  predicted_translation = index_to_latscr[predicted_class_index]
  print(f'Translated Baybayin to English: {predicted_translation}')