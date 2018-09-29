# -*- coding: utf8 -*-
import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.layers import LSTM, Embedding
from keras.layers import TimeDistributed
from keras.models import Model, Sequential

# Inception模型（图像分类）

input_img = Input(shape=(256, 256, 3))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

# 残差网络 卷积层的残差连接 （图像分类）

x = Input(shape=(256, 256, 3))

y = Conv2D(3, (3, 3), padding='same')(x)

z = keras.layers.add([x, y])

# 共享视觉模型 （MNIST数字识别中判断数字是否相同）

digit_input = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

digit_a = Input(shape=(27, 27, 1))
digit_b = Input(shape=(27, 27, 1))

# 共享vision_model，如weights等
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = keras.layers.concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], out)

# 视觉问答模型（对一幅图片提问，返回关于该图片的答案）

# vision_model:获得图片的vector表示
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D(2, 2))
vision_model.add(Flatten())

image_input = Input(shape=(224, 224, 3))
encoded_image = vision_model(image_input)

# 获得问题的vector表示
question_input = Input(shape=(100, ), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

merged = keras.layers.concatenate([encoded_question, encoded_image])

output = Dense(1000, activation='softmax')(merged)  # 有1000个词选择

vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# 视频问答模型

video_input = Input(shape=(100, 224, 224, 3))

encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # 将图片模型转化成视频模型，返回vector序列
encoded_video = LSTM(256)(encoded_frame_sequence)

question_encoder = Model(inputs=question_input, outputs=encoded_question)

video_question_input = Input(shape=(100, ), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

merged = keras.layers.concatenate([encoded_video, encoded_question])

output = Dense(1000, activation='softmax')(merged)

video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)









