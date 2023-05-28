import tensorflow as tf
from tensorflow.keras import layers


# # Функция активации ReLU
# def relu(x):
#     return tf.keras.activations.relu(x)
#
#
# # Функция активации Sigmoid
# def sigmoid(x):
#     return tf.keras.activations.sigmoid(x)
#
#
# # Класс U-Net
# class UNet(tf.keras.Model):
#     def __init__(self):
#         super(UNet, self).__init__()
#
#         # Сверточные слои
#         self.conv1 = layers.Conv2D(32, 3, activation=relu, padding='same')
#         self.conv2 = layers.Conv2D(32, 3, activation=relu, padding='same')
#         self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
#         self.conv3 = layers.Conv2D(64, 3, activation=relu, padding='same')
#         self.conv4 = layers.Conv2D(64, 3, activation=relu, padding='same')
#         self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
#         self.conv5 = layers.Conv2D(64, 3, activation=relu, padding='same')
#         self.conv6 = layers.Conv2D(64, 3, activation=relu, padding='same')
#         self.up1 = layers.UpSampling2D(size=(2, 2))
#         self.conv7 = layers.Conv2D(64, 3, activation=relu, padding='same')
#         self.conv8 = layers.Conv2D(64, 3, activation=relu, padding='same')
#         self.up2 = layers.UpSampling2D(size=(2, 2))
#         self.conv9 = layers.Conv2D(32, 3, activation=relu, padding='same')
#         self.conv10 = layers.Conv2D(1, 3, activation=relu, padding='same')
#
#     def call(self, inputs):
#         # Прямой проход через нейронную сеть
#         x1 = self.conv1(inputs)
#         x1 = self.conv2(x1)
#         x2 = self.pool1(x1)
#         x2 = self.conv3(x2)
#         x2 = self.conv4(x2)
#         x3 = self.pool2(x2)
#         x3 = self.conv5(x3)
#         x3 = self.conv6(x3)
#         x4 = self.up1(x3)
#         x4 = tf.concat([x4, x2], axis=-1)
#         x4 = self.conv7(x4)
#         x4 = self.conv8(x4)
#         x5 = self.up2(x4)
#         x5 = tf.concat([x5, x1], axis=-1)
#         x5 = self.conv9(x5)
#         output = self.conv10(x5)
#         output = sigmoid(output)  # Выходной слой сигмоида для вероятностей
#
#         return output


def unet_model(input_shape):
    # Определение архитектуры модели U-Net
    inputs = tf.keras.Input(shape=input_shape)
    # Свёрточные слои
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Повышающие дискретизации
    up5 = layers.Conv2D(64, 3, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv4))
    merge5 = layers.concatenate([conv3, up5], axis=3)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)

    up6 = layers.Conv2D(64, 3, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv5))
    merge6 = layers.concatenate([conv2, up6], axis=3)
    conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2D(32, 3, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv1, up7], axis=3)
    conv7 = layers.Conv2D(32, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv7)

    # Слой сегментации
    segmentation = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(conv7)

    model = tf.keras.Model(inputs=inputs, outputs=segmentation)

    return model