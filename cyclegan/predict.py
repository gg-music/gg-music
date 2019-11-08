import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import numpy as np
from .helpers.utils import make_dirs, get_file_list, check_rawdata_exists
from .helpers.signal import (preprocessing_fn, inverse_fn, write_audio)
from .helpers.plot import plot_heat_map
from .settings import (DEFAULT_SAMPLING_RATE, MODEL_ROOT_PATH,
                       WAVS_TO_PREDICT_ROOT_PATH)
from random import shuffle


def predict(model, spec_format, input_filename, output_filename):

    mag, phase = preprocessing_fn(input_filename, spec_format)
    mag = mag[np.newaxis, :]

    mag = model.predict(mag)

    audio_out = inverse_fn(mag, phase, spec_format)

    make_dirs(os.path.dirname(output_filename))
    write_audio(output_filename, audio_out, DEFAULT_SAMPLING_RATE)


def load_model(model_path, n_epoch):
    import tensorflow as tf
    from .model_settings import generator_g, generator_f

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                               generator_f=generator_f)

    ckpt_manager = tf.train.CheckpointManager(ckpt, model_path, max_to_keep=100)
    last_epoch = len(ckpt_manager.checkpoints)

    if n_epoch:
        epoch = n_epoch
        ckpt.restore(ckpt_manager.checkpoints[epoch - 1]).expect_partial()
        print('Checkpoint epoch {} restored!!'.format(epoch))
    else:
        epoch = last_epoch
        ckpt.restore(ckpt_manager.checkpoints[epoch - 1]).expect_partial()
        print('Latest checkpoint epoch {} restored!!'.format(epoch))

    models = {'g': generator_g, 'f': generator_f}

    return ckpt, ckpt_manager, models, epoch


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True)
    ap.add_argument('-e', '--epoch', required=False, type=int)
    ap.add_argument('-cqt',
                    '--spectrum',
                    required=False,
                    default=False,
                    action='store_true',
                    help='convert to cqt, default is stft')
    ap.add_argument('-x', required=True, help='convert from', type=str)
    ap.add_argument('-y', required=True, help='convert to', type=str)
    ap.add_argument('-n',
                    '--n_samples',
                    required=False,
                    default=1,
                    help='n_samples to predict each instrument',
                    type=int)
    args = ap.parse_args()

    MODEL_PATH = os.path.join(MODEL_ROOT_PATH, args.model)
    SAVE_WAV_PATH = os.path.join(MODEL_PATH, 'wav')
    instrument_x_wav_path = os.path.join(WAVS_TO_PREDICT_ROOT_PATH, args.x)
    instrument_y_wav_path = os.path.join(WAVS_TO_PREDICT_ROOT_PATH, args.y)

    check_rawdata_exists(instrument_x_wav_path, instrument_y_wav_path)

    input_files_x = get_file_list(instrument_x_wav_path)
    input_files_y = get_file_list(instrument_y_wav_path)
    shuffle(input_files_x)
    shuffle(input_files_y)
    input_files_x = input_files_x[:args.n_samples]
    input_files_y = input_files_y[:args.n_samples]

    ckpt, ckpt_manager, models, epoch = load_model(MODEL_PATH, args.epoch)

    input_files = {'g': input_files_x, 'f': input_files_y}

    for model, wavs in input_files.items():
        for wav in wavs:
            output_file = os.path.join(
                SAVE_WAV_PATH, args.model + "-" +
                               os.path.basename(ckpt_manager.checkpoints[epoch - 1]) + "-" +
                               os.path.basename(wav))

            predict(models[model], args.spectrum, wav, output_file)
            print('Prediction saved in', output_file)
