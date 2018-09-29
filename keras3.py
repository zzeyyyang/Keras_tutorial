# -*- coding: utf8 -*-
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import numpy as np
from keras.utils import plot_model


# 路透社新闻主题分类数据（新闻单词已化成数字）

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)  # 数据集默认划分是8:2


# 数据向量化
def vetorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# 标签向量化
def to_one_hot(labels, dimension=46):  # 46个类别
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


x_train = vetorize_sequences(train_data)
y_train = to_one_hot(train_labels)

x_train_used = x_train[1000:]
y_train_used = y_train[1000:]

x_validation = x_train[:1000]
y_validation = y_train[:1000]

x_test = vetorize_sequences(test_data)
y_test = to_one_hot(test_labels)

# 基于多层感知机的softmax多分类

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(10000,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(46, activation='softmax'))  # 输出结果为64维

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # 随机梯度下降，decay为每次更新时lr衰减量

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_used, y_train_used, epochs=20, batch_size=128, validation_data=(x_validation, y_validation))

plot_model(model, to_file='model.png')  # 可视化模型结构

score = model.evaluate(x_test, y_test, batch_size=128)  # loss: score[0], accuracy: score[1]

print '/n'
print score






