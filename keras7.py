# -*- coding: utf8 -*-
import keras
from keras.layers import Input, Dense, LSTM, Embedding
from keras.models import Model
from keras.utils import to_categorical
import numpy as np

# Functional Model

# 全连接网络

data = np.random.random((10000, 100))
one_hot_labels = to_categorical(np.random.randint(10, size=(10000, 1)), num_classes=10)

inputs = Input(shape=(100, ))

x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, one_hot_labels, epochs=20, batch_size=32)

# 多输入和多输出模型

headline_data = np.random.randint(0, 1, size=(10000, 100))
additional_data = np.random.randint(2, 6, size=(10000, 5))
labels = np.random.randint(2, size=(10000, 1))

main_input = Input(shape=(100, ), dtype='int32', name='main_input')
# main_input接收新闻，即一个位于1到10000之间（即我们的字典有10000个词）的整数的序列（每个整数编码了一个词），这个序列有100个单词

x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

lstm_out = LSTM(32)(x)

auxiliary_output = Dense(1, activation='sigmoid',  # 额外损失
                         name='aux_output')(lstm_out)  # 使得即使在主损失很高的情况下，LSTM和Embedding层也可以平滑的训练

auxiliary_input = Input(shape=(5, ), name='aux_input')  # auxiliary_input接收日期等
x = keras.layers.concatenate([lstm_out, auxiliary_input])  # 将LSTM与额外的输入数据串联起来组成输入

x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])

model.fit([headline_data, additional_data], [labels, labels],
          epochs=20, batch_size=32)

'''
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# And trained it via:
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
'''

# 共享层

tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))

shared_lstm = LSTM(64)

encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# 查看隐藏层输出
assert shared_lstm.get_output_at(0) == encoded_a
assert shared_lstm.get_output_at(1) == encoded_b

merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

predictions = Dense(1, activation='sigmoid')(merged_vector)

model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)








