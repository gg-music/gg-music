import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Input, ZeroPadding2D, MaxPooling2D, Conv2D, Dropout, concatenate, Cropping2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np

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
    nb_filter = [64, 128, 256, 512]
    inp = Input(shape=[img_rows, img_cols, 1], name='input_image')
    tar = Input(shape=[img_rows, img_cols, 1], name='target_image')

    x = concatenate([inp, tar])

    # img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    # z_pad = ZeroPadding2D(padding_size)(img_input)

    conv1_1 = standard_unit(x, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv4_1)

    zero_pad1 = ZeroPadding2D()(pool4)

    conv = Conv2D(512,
                  4,
                  strides=1,
                  kernel_initializer=initializer,
                  use_bias=False)(zero_pad1)

    batchnorm1 = BatchNormalization()(conv)

    leaky_relu = LeakyReLU()(batchnorm1)

    last = Conv2D(1, 4, strides=1, kernel_initializer=initializer)(leaky_relu)

    return Model(inputs=[inp, tar], outputs=last)


if __name__ == '__main__':
    inp = np.load('/home/gtzan/data/gan_preprocessing/guitar1/guitar1-000.npy')
    tar = np.load('/home/gtzan/data/gan_preprocessing/guitar1/guitar1-000.npy')
    model = get_model(inp.shape[0], inp.shape[1])
    print(model.summary())
