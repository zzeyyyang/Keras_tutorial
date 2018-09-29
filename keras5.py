# -*- coding: utf8 -*-
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical

# cifar10图像分类数据集

(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

x_train = train_data.astype('float32') / 255
y_train = to_categorical(train_labels, num_classes=10)

x_train_used = x_train[5000:]
y_train_used = y_train[5000:]

x_validation = x_train[:5000]
y_validation = y_train[:5000]

x_test = test_data.astype('float32') / 255
y_test = to_categorical(test_labels, num_classes=10)

# 类似VGG的卷积神经网络

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(x_train.shape[1:])))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())  # 将卷积池化后的2维结果变成1维
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train_used, y_train_used, batch_size=32, epochs=10, validation_data=(x_validation, y_validation))
score = model.evaluate(x_test, y_test, batch_size=32)

print '/n'
print score
