import os
import numpy as np
import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
from cyclegan.settings import DEFAULT_SAMPLING_RATE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cyclegan.helpers.signal import preprocessing_fn, inverse_fn, mag_inverse, log_fq
from cyclegan.helpers.plot import plot_heat_map
from cyclegan.settings import DEFAULT_SAMPLING_RATE
import tensorflow as tf


def plot(mag, title):
    mag = np.repeat(mag[:, :, np.newaxis], 3, axis=-1)
    mag = mag[np.newaxis, :]
    plot_heat_map(mag, title=title)


TEST_DIR = '/home/gtzan/ssd/test'
inp = '/home/gtzan/data/gan_preprocessing/wav/sax/sax-0334.wav'

mag, phase = preprocessing_fn(inp, spec_type='harm')
print(mag.shape)
plot_heat_map(mag, title='harm_x')

mag = log_fq(mag)
mag = mag[0, :, :, 0]

sos = signal.butter(1, 2, 'hp', fs=512, output='sos')


harm_bank = np.zeros((256,256,256))
for base in range(256):
    for harm in range(256):
        f = base * harm
        if f < 256:
            harm_bank[base, harm, f] = 1

masked = np.zeros((256,256,256))
for t in range(256):
    for i, base in enumerate(harm_bank):
        for j, harm in enumerate(base):
            masked[t, i, j] = np.dot(mag[t], harm)

plot(masked[7], title="mask")
