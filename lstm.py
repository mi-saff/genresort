#Inspired by https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

import numpy as np
import rnn_embedding
import generate_data
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.preprocessing import sequence

np.random.seed(7)

#Load in the data with a special function that embeds it all as unique
#integers corresponding to the position in the GloVE embedding
#dictionary.  This will help us train our embedding layer, albiet at the
#expense of a massive set of parameters.

X_train, y_train = rnn_embedding.return_stuff("bigdata")
X_test, y_test = rnn_embedding.return_stuff("bigtest")

#Size of GloVE vocabulary
vocab = 400000

#The maximum paragraph length we will pad for
max_par_len = 200

#Pads the data
X_train = sequence.pad_sequences(X_train, maxlen=max_par_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_par_len)

#Size of our embedding layer
embed_size = 100
model = keras.models.Sequential()

#Implementation of the model
model.add(keras.layers.embeddings.Embedding(vocab, embed_size, input_length=max_par_len))
model.add(keras.layers.convolutional.Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(10)))
model.add(Dropout(0.1))
model.add(Dense(5, activation='softmax'))
model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=24, batch_size=500)
scores = model.evaluate(X_test, y_test, verbose=0)
#Prints out our test accuracy
print("Accuracy: %.2f%%" % (scores[1]*100))
