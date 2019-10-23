import numpy as np
import math
import os
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from librosa.util import normalize

from .utils import unet_padding_size
from .signal import amplitude_to_db, slice_magnitude


class GanSequence(Sequence):

    def __init__(self, file_list, batch_size=32, shuffle=False):
        self.file_list = file_list
        self.dim = self.get_dim()
        self.pad_size = ((32, 32),
                         unet_padding_size(self.dim[1], pool_size=2, layers=5))
        self.batch_size = batch_size
        self.input_shape = self.get_input_shape()
        self.batch_dim = self.get_batch_dim()
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_list))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.file_list) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        temp_file_list = [self.file_list[k] for k in indexes]
        data = self.get_data(temp_file_list)
        return data

    def get_dim(self):
        x = np.load(self.file_list[0])
        return x.shape

    def get_batch_dim(self):
        return self.batch_size, self.input_shape[0], self.input_shape[1], 3

    def get_input_shape(self):
        return (self.dim[0] + self.pad_size[0][0] + self.pad_size[0][1],
                self.dim[1] + self.pad_size[1][0] + self.pad_size[1][1], 3)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @staticmethod
    def reshape(img):
        img = tf.reshape(img, shape=(1, img.shape[0], img.shape[1], 1))
        return img

    def get_data(self, temp_file_list):
        for i, ID in enumerate(temp_file_list):
            mag = np.load(ID)
            if np.max(mag) == 0:
                os.remove(ID)
                print("\nremove zero file: " + ID + "\n")
            mag_db = amplitude_to_db(mag)
            mag_db = (mag_db * 2) - 1
            X = np.pad(mag_db, self.pad_size)
            X = np.repeat(X.reshape(
                (self.input_shape[0], self.input_shape[1], 1)),
                          3,
                          axis=2)
            X = X[np.newaxis, :]

        return X
