import os
import tensorflow as tf
from gtzan.data_generator import GanSequence
from gtzan import signal
from cyclegan.model_settings import *
from cyclegan.settings import DEFAULT_SAMPLING_RATE, CHECKPOINT_PATH
import numpy as np

def predict(model, input_filename, output_filename, crop_size):
    audio = signal.load_audio(input_filename, sr=DEFAULT_SAMPLING_RATE)

    mag, phase = signal.to_stft(audio)
    mag_db = signal.amplitude_to_db(mag)

    mag_sliced = signal.slice_magnitude(mag_db, mag.shape[1])
    mag_sliced = (mag_sliced * 2) - 1

    X = np.repeat(mag_sliced.reshape((mag_sliced.shape[0], mag_sliced.shape[1], 1)), 3, axis=2)
    X = X[np.newaxis, :]

    prediction = model.predict(X)
    prediction = (prediction + 1) / 2

    mag_db = signal.join_magnitude_slices(prediction, phase.shape, crop_size)
    mag = signal.db_to_amplitude(mag_db)
    audio_out = signal.inverse_transform(mag, phase)
    signal.write_audio(output_filename, audio_out)


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
                                              max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
    predict(ckpt.generator_g,
            '/home/gtzan/data/gan/wav/sounds/piano1/piano1-484.wav',
            'args.output_path', ((32, 32), (40, 41)))
    print('Prediction saved in', args.output)
