import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Input, ZeroPadding2D, MaxPooling2D, Conv2D, Dropout, concatenate, \
    Cropping2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np

from gtzan.utils import unet_padding_size

ACTIVATION_FN = 'relu'
DROPOUT_RATE = 0.5
BN_AXIS = 3


def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    x = Conv2D(nb_filter, (kernel_size, kernel_size),
               name='conv' + stage + '_1',
               kernel_initializer='he_normal',
               padding='same',
               kernel_regularizer=l2(1e-4))(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(DROPOUT_RATE, name='dp' + stage + '_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size),
               name='conv' + stage + '_2',
               kernel_initializer='he_normal',
               padding='same',
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(DROPOUT_RATE, name='dp' + stage + '_2')(x)

    return x


def get_model(img_rows, img_cols):
    initializer = tf.random_normal_initializer(0., 0.02)
    nb_filter = [64, 128, 256, 512, 512]

    # Handle Dimension Ordering for different backends
    img_input = Input(shape=(img_rows, img_cols, 3), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2),
                            strides=(2, 2),
                            name='up42',
                            padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=BN_AXIS)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2),
                            strides=(2, 2),
                            name='up33',
                            padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=BN_AXIS)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2),
                            strides=(2, 2),
                            name='up24',
                            padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=BN_AXIS)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    unet_output = Conv2DTranspose(3, (2, 2),
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')(conv2_4)

    model = Model(inputs=img_input, outputs=unet_output)

    return model


if __name__ == '__main__':
    inp = np.load('/home/gtzan/data/gan_preprocessing/guitar1/guitar1-060.npy')
    inp = inp.reshape(256, 431, 1)

    PAD_SIZE = ((0, 0), unet_padding_size(inp.shape[1], pool_size=2, layers=8))
    IPT_SHAPE = [inp.shape[0], inp.shape[1] + PAD_SIZE[1][0] + PAD_SIZE[1][1]]
    model = get_model(IPT_SHAPE[0], IPT_SHAPE[1])
    print(model.summary())
