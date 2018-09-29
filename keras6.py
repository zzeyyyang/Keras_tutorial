# -*- coding:utf-8-*-
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import EarlyStopping

# imdb数据集

max_features = 20000  # 词汇表大小
max_len = 80  # 每个句子包含单词数
time_steps = 80  # 时间步（每个时刻输入的单词数量）

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(train_data, maxlen=max_len)  # 将句子剪裁成相同序列，不足补0
y_train = train_labels

x_test = sequence.pad_sequences(test_data)
y_test = test_labels

# 提前结束训练
early_stopping = EarlyStopping(monitor='val_loss', patience=2)  # monitor为监测的指标，若不在变化，执行patience个epoch后结束

# lstm分类

model1 = Sequential()
model1.add(Embedding(input_dim=max_features, output_dim=256))
model1.add(LSTM(128))
model1.add(Dropout(0.5))
model1.add(Dense(1, activation='sigmoid'))

model1.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model1.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

score1 = model1.evaluate(x_test, y_test, batch_size=32)

print '/n'
print score1

# 使用1D卷积的序列分类

model2 = Sequential()

model2.add(Embedding(input_dim=max_features, output_dim=256))  # 将max_features降维到256
model2.add(Conv1D(64, 3, activation='relu', input_shape=(max_len, max_features)))  # Con1D常用于NLP卷积
model2.add(Conv1D(64, 3, activation='relu'))
model2.add(MaxPooling1D(3))
model2.add(Conv1D(128, 3, activation='relu'))
model2.add(Conv1D(128, 3))
model2.add(GlobalAveragePooling1D())  # 平均池化
model2.add(Dropout(0.5))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model2.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

score2 = model2.evaluate(x_test, y_test, batch_size=32)

print '/n'
print score2

# 用于序列分类的栈式LSTM

model3 = Sequential()

model3.add(Embedding(input_dim=max_features, output_dim=256))
model3.add(LSTM(32,
                return_sequences=True,  # return_sequences为True，返回LSTM所有时间步序列
                input_shape=(time_steps, 256)))
model3.add(LSTM(32, return_sequences=True))
model3.add(LSTM(32))
model3.add(Dense(1, activation='sigmoid'))

model3.compile(optimizer='rmsprop',
               loss='binary_crossentropy',
               metrics=['accuracy'])

model3.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

score3 = model3.evaluate(x_test, y_test, batch_size=32)

print '/n'
print score3

# 采用stateful LSTM的相同模型
# 在处理过一个batch的训练数据后，其内部状态会被作为下一个batch的训练数据的初始状态，使可以在合理的计算复杂度内处理较长序列

model4 = Sequential()
model4.add(Embedding(input_dim=max_features, output_dim=256))
model4.add(LSTM(32, return_sequences=True, stateful=True, batch_input_shape=(32, time_steps, 256)))
model4.add(LSTM(32, return_sequences=True, stateful=True))
model4.add(LSTM(32, stateful=True))
model4.add(Dense(10, activation='softmax'))

model4.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model4.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

score4 = model3.evaluate(x_test, y_test, batch_size=32)

print '/n'
print score4




