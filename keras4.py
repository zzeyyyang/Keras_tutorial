# -*- coding: utf8 -*-
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np


# imdb情感分类数据集

(train_data, train_labels), (test_data, teat_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequences(train_data)

x_train_used = x_train[2500:]
y_train_used = train_labels[2500:]

x_validation = x_train[:2500]
y_validation = train_labels[:2500]

x_test = vectorize_sequences(test_data)
y_test = teat_labels

# MLP的二分类

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(10000, )))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train_used, y_train_used, epochs=20, batch_size=128, validation_data=(x_validation, y_validation))

score = model.evaluate(x_test, y_test, batch_size=128)

print '\n'
print score

