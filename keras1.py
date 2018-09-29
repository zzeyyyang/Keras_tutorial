# -*- coding:utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.backend as K


# 构建模型

model = Sequential()

model.add(Dense(32, input_dim=724))
model.add(Activation('relu'))

# 编译

# 多分类问题
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 二分类问题
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 均方根作为误差回归问题
model.compile(optimizer='rmsprop',
              loss='mse')


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])

# 训练
model.fit()




