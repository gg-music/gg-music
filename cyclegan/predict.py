import os
import argparse
import numpy as np
import tensorflow as tf
from .helpers import signal
from .helpers.utils import preprocessing_fn, make_dirs
from .model_settings import *
from .settings import DEFAULT_SAMPLING_RATE, PAD_SIZE, MODEL_ROOT_PATH


def predict(model, input_filename, output_filename):
    mag, phase = preprocessing_fn(input_filename,
                                  spec_format=signal.to_stft)
    mag = mag[np.newaxis, :]

    prediction = model.predict(mag)
    prediction = (prediction + 1) / 2

    mag = signal.crop(prediction, PAD_SIZE)
    mag = signal.join_magnitude_slices(mag, target_shape=phase.shape)
    mag = signal.db_to_amplitude(mag)
    audio_out = signal.inverse_stft(mag, phase)

    make_dirs(os.path.dirname(output_filename))
    signal.write_audio(output_filename, audio_out, DEFAULT_SAMPLING_RATE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True)
    args = ap.parse_args()

    MODEL_PATH = os.path.join(MODEL_ROOT_PATH, args.model)
    WAV_PATH = os.path.join(MODEL_PATH, 'wav')

    ckpt = tf.train.Checkpoint(
        generator_g=generator_g,
        generator_f=generator_f,
        discriminator_x=discriminator_x,
        discriminator_y=discriminator_y,
        generator_g_optimizer=generator_g_optimizer,
        generator_f_optimizer=generator_f_optimizer,
        discriminator_x_optimizer=discriminator_x_optimizer,
        discriminator_y_optimizer=discriminator_y_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, MODEL_PATH, max_to_keep=100)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    last_epoch = len(ckpt_manager.checkpoints)
    print('Latest checkpoint epoch {} restored!!'.format(last_epoch))

    input_file = [['/home/gtzan/data/gan/wav/sounds/cello/cello-600.wav', ckpt.generator_g],
                  ['/home/gtzan/data/gan/wav/sounds/sax/sax-600.wav', ckpt.generator_f]]

    for wav, model in input_file:
        output_file = os.path.join(WAV_PATH,
                                   os.path.basename(ckpt_manager.latest_checkpoint)
                                   + "-" + os.path.basename(wav))

        predict(model, wav, output_file)
        print('Prediction saved in', output_file)
