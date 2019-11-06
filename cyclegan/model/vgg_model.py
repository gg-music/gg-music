import tensorflow as tf
from tensorflow.keras.layers import Input, LeakyReLU, ZeroPadding2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization, concatenate, Conv2D, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from .pix2pix import InstanceNormalization


def vgg16_model(input_shape=(None, None, 3), norm_type='batchnorm', target=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = Input(shape=input_shape, name='input_image')
    x = inp

    if target:
        tar = Input(shape=input_shape, name='target_image')
        x = concatenate([inp, tar], axis=1)  # (bs, 256, 256, channels*2)

    vgg16 = VGG16(include_top=False, weights='imagenet',
                  input_tensor=x)

    conv1 = Conv2D(512, (8, 1), strides=1, padding='same',
                   kernel_initializer=initializer)(vgg16.layers[-2].output)

    conv2 = Conv2D(512, (8, 1), strides=1, activation='relu',
                   kernel_initializer=initializer)(conv1)
    conv3 = Conv2D(512, (8, 1), strides=1, activation='relu',
                   kernel_initializer=initializer)(conv2)
    drop1 = Dropout(0.1)(conv3)
    conv4 = Conv2D(512, (8, 1), strides=1, activation='relu',
                   kernel_initializer=initializer)(drop1)
    conv5 = Conv2D(512, (8, 1), strides=1, activation='relu',
                   kernel_initializer=initializer)(conv4)
    drop2 = Dropout(0.1)(conv5)

    if norm_type.lower() == 'batchnorm':
        norm1 = BatchNormalization()(drop2)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(drop2)

    leaky_relu = LeakyReLU()(norm1)

    last = tf.keras.layers.Conv2D(1, (4, 1), strides=1,
        kernel_initializer=initializer)(leaky_relu)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)


if __name__ == '__main__':
    model = vgg16_model(input_shape=(512, 256, 3))
    model.summary()
