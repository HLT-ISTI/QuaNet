from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Bidirectional
from keras.datasets import imdb

max_features = 5000
maxlen = 80
batch_size = 200
embedding_dims = 100
hidden_dims = 250
epochs = 10

### IMDB is a too simple dataset, using it just to see it this idea works

print('Loading data...')
# start_char is set to 2 because 0 is for padding and 1 is for sequence restart signal
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, start_char=2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Simple LSTM

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embedding_dims))
model.add(LSTM(hidden_dims, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# Bidirectional LSTM

print('Build model...')
bi_model = Sequential()
bi_model.add(Embedding(max_features, embedding_dims))
bi_model.add(Bidirectional(LSTM(hidden_dims, dropout=0.2, recurrent_dropout=0.2)))
bi_model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
bi_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

bi_model.summary()

print('Train...')
bi_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = bi_model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# Repeated sequence/simple LSTM

import numpy as np
# using the reserved id 1 as the signal of sequence restart
d_x_train = np.hstack([x_train,np.expand_dims(np.asarray([1]*x_train.shape[0]),1),x_train])
d_x_test = np.hstack([x_test,np.expand_dims(np.asarray([1]*x_test.shape[0]),1),x_test])
d_x_train.shape, d_x_test.shape

print('Build model...')
d_model = Sequential()
d_model.add(Embedding(max_features, embedding_dims))
d_model.add(LSTM(hidden_dims, dropout=0.2, recurrent_dropout=0.2))
d_model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
d_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

d_model.summary()

print('Train...')
d_model.fit(d_x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(d_x_test, y_test))
score, acc = d_model.evaluate(d_x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)