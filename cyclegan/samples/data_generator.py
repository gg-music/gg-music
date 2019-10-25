import numpy as np
import math
import os
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from cyclegan.settings import PAD_SIZE

class GanSequence(Sequence):

    def __init__(self, file_list, batch_size=32, shuffle=False):
        self.file_list = file_list
        self.dim = self.get_dim()
        self.pad_size = PAD_SIZE
        self.batch_size = batch_size
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

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @staticmethod
    def reshape(img):
        img = tf.reshape(img, shape=(1, img.shape[0], img.shape[1], 1))
        return img

    def get_data(self, temp_file_list):
        batch = []
        for i, file in enumerate(temp_file_list):
            batch.append(np.load(file))
        return np.array(batch)
