# -*- coding:utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np

# Sequential Model

# 二分类

data1 = np.random.random((1000, 100))  # 产生0-1之间的1000条100维的data
labels1 = np.random.randint(2, size=(1000, 1))  # 产生0或1的1000个label

model1 = Sequential()
model1.add(Dense(32, activation='relu', input_dim=100))
model1.add(Dense(1, activation='sigmoid'))

model1.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model1.fit(data1, labels1, epochs=10, batch_size=32)

# 多分类

data2 = np.random.random((1000, 100))
labels2 = np.random.randint(10, size=(1000, 1))
one_hot_labels = to_categorical(labels2, num_classes=10)  # label转换成独热编码表示

model2 = Sequential()
model2.add(Dense(32, activation='relu', input_dim=100))
model2.add(Dense(10, activation='softmax'))

model2.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

model2.fit(data2, one_hot_labels, epochs=10, batch_size=32)





