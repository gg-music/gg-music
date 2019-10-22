import os
import tensorflow as tf
from gtzan.data_generator import GanSequence
from gtzan import signal
from cyclegan.model_settings import *
from cyclegan.settings import DEFAULT_SAMPLING_RATE, CHECKPOINT_PATH
import numpy as np
from gtzan.utils import unet_padding_size
from librosa.util import normalize


def get_input_shape(dim, pad_size):
    return (dim[0] + pad_size[0][0] + pad_size[0][1],
            dim[1] + pad_size[1][0] + pad_size[1][1], 3)


def get_input_shape_sliced(dim, pad_size):
    return (1, dim[1] + pad_size[1][0] + pad_size[1][1],
            dim[2] + pad_size[2][0] + pad_size[2][1], 3)


def get_pad_size(dim):
    return ((32, 32), unet_padding_size(dim[1], pool_size=2, layers=8))


def get_pad_size_sliced(dim):
    return ((0, 0), (32, 32), unet_padding_size(dim[2], pool_size=2,
                                                layers=8), (0, 0))


def predict(model, input_filename, output_filename):
    audio = signal.load_audio(input_filename, sr=DEFAULT_SAMPLING_RATE)
    mag, phase = signal.to_stft(audio)
    print('raw.shape', mag.shape)
    print('raw.phase.shape', phase.shape)
    mag_db = signal.amplitude_to_db(mag)
    mag_sliced = signal.slice_magnitude(mag_db, mag.shape[1])
    mag_sliced = (mag_sliced * 2) - 1

    # reshape and pad for model prediction
    pad_size = get_pad_size_sliced(mag_sliced.shape)
    input_shape = get_input_shape_sliced(mag_sliced.shape, pad_size)
    X = np.pad(mag_sliced, pad_size)
    X = np.repeat(X.reshape((input_shape[1], input_shape[2], 1)), 3, axis=2)
    X = X[np.newaxis, :]

    prediction = model.predict(X)
    prediction = (prediction + 1) / 2
    # remove padding
    unpadded = signal.unpad_to_raw(prediction, pad_size)

    mag_db = signal.join_magnitude_slices(unpadded, phase.shape, pad_size)
    mag = signal.db_to_amplitude(mag_db)
    audio_out = signal.inverse_transform(mag, phase)
    signal.write_audio(output_filename, audio_out, DEFAULT_SAMPLING_RATE)


if __name__ == "__main__":
    # Enable mixed precision
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    ckpt = tf.train.Checkpoint(
        generator_g=generator_g,
        generator_f=generator_f,
        discriminator_x=discriminator_x,
        discriminator_y=discriminator_y,
        generator_g_optimizer=generator_g_optimizer,
        generator_f_optimizer=generator_f_optimizer,
        discriminator_x_optimizer=discriminator_x_optimizer,
        discriminator_y_optimizer=discriminator_y_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              CHECKPOINT_PATH,
                                              max_to_keep=100)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

    input_file = '/home/gtzan/data/gan/wav/sounds/piano1/piano1-166.wav'
    output_file = f'/home/gtzan/data/gan_output/piano_to_guitar/{input_file.split("/")[-1]}'
    # predict_data_gen = PredictSequence()
    predict(ckpt.generator_g, input_file, output_file)
    print('Prediction saved in', output_file)

    input_file = '/home/gtzan/data/gan/wav/sounds/guitar2/guitar2-333.wav'
    output_file = f'/home/gtzan/data/gan_output/guitar_to_piano/{input_file.split("/")[-1]}'
    # predict_data_gen = PredictSequence()
    predict(ckpt.generator_f, input_file, output_file)
    print('Prediction saved in', output_file)
