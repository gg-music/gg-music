from abc import abstractmethod
import numpy as np
import math
import os
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from librosa.util import normalize

from gtzan.utils import unet_padding_size
from gtzan.signal import amplitude_to_db, slice_magnitude


class BaseSequence(Sequence):

    def __init__(self, file_list, batch_size=32, shuffle=False):
        self.file_list = file_list
        self.dim = self.get_dim()
        self.batch_size = batch_size
        self.batch_dim = self.get_batch_dim()
        self.input_shape = self.get_input_shape()
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
        x = np.load(self.file_list[0][0])
        return x.shape

    def get_batch_dim(self):
        return self.batch_size * self.dim[0], self.dim[1], self.dim[
            2], self.dim[3]

    def get_input_shape(self):
        return self.dim[1], self.dim[2], 3

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @abstractmethod
    def get_data(self, temp_file_list):
        raise NotImplementedError


class DataSequence(BaseSequence):

    def get_data(self, temp_file_list):
        spec_file = np.zeros(self.batch_dim)
        category = np.zeros((self.batch_size * self.dim[0]))
        for i, line in enumerate(temp_file_list):
            ID = line[0]
            cat = line[1]
            X = np.load(ID)

            try:
                for j in range(X.shape[0]):
                    spec_file[i * X.shape[0] + j,] = X[j]
                    category[i * X.shape[0] + j,] = cat

            except Exception as E:
                os.remove(ID)
                print("\nremove corrupt file: " + ID + "\n")
                continue

        X_stack = np.squeeze(np.stack((spec_file,) * 3, axis=-1))
        return X_stack, category


class PredictSequence(BaseSequence):

    def get_data(self, temp_file_list):
        spec_file = np.empty(self.batch_dim)
        for i, line in enumerate(temp_file_list):
            ID = line[0]
            X = np.load(ID)

            try:
                for j in range(X.shape[0]):
                    spec_file[i * X.shape[0] + j,] = X[j]

            except Exception as E:
                print("\ncorrupt file: " + ID + "\n")
                continue

        X_stack = np.squeeze(np.stack((spec_file,) * 3, axis=-1))
        return X_stack
