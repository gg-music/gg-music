import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import clone_model
from .helpers.utils import make_dirs, load_model, get_file_list, check_rawdata_exists
from .model.model_settings import generator_g, generator_f
from .helpers.signal import predict, write_audio
from .settings import (DEFAULT_SAMPLING_RATE, MODEL_ROOT_PATH,
                       WAVS_TO_PREDICT_ROOT_PATH)
from random import shuffle

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True)
    ap.add_argument('-m2', '--model2', required=False)
    ap.add_argument('-e', '--epoch', required=False, type=int)
    ap.add_argument('-e2', '--epoch2', required=False, type=int)
    ap.add_argument('-x', required=True, help='convert from', type=str)
    ap.add_argument('-y', required=True, help='convert to', type=str)
    ap.add_argument('-n',
                    '--n_samples',
                    required=False,
                    default=1,
                    help='n_samples to predict each instrument',
                    type=int)
    args = ap.parse_args()

    instrument_x_wav_path = os.path.join(WAVS_TO_PREDICT_ROOT_PATH, args.x)
    instrument_y_wav_path = os.path.join(WAVS_TO_PREDICT_ROOT_PATH, args.y)

    check_rawdata_exists(instrument_x_wav_path, instrument_y_wav_path)

    input_files_x = get_file_list(instrument_x_wav_path)
    input_files_y = get_file_list(instrument_y_wav_path)
    shuffle(input_files_x)
    shuffle(input_files_y)
    input_files_x = input_files_x[:args.n_samples]
    input_files_y = input_files_y[:args.n_samples]

    MODEL_PATH = os.path.join(MODEL_ROOT_PATH, args.model)
    SAVE_WAV_PATH = os.path.join(MODEL_PATH, 'wav')
    g1 = clone_model(generator_g)
    f1 = clone_model(generator_f)
    ckpt = tf.train.Checkpoint(generator_g=g1,
                               generator_f=f1)
    ckpt_manager = tf.train.CheckpointManager(ckpt, MODEL_PATH, max_to_keep=100)
    load_model(args.epoch, ckpt, ckpt_manager)
    models = {'g': g1, 'f': f1}

    if args.model2:
        MODEL2_PATH = os.path.join(MODEL_ROOT_PATH, args.model2)
        g2 = clone_model(generator_g)
        f2 = clone_model(generator_f)
        ckpt2 = tf.train.Checkpoint(generator_g=g2,
                                    generator_f=f2)
        ckpt_manager2 = tf.train.CheckpointManager(ckpt2, MODEL2_PATH, max_to_keep=100)
        load_model(args.epoch2, ckpt2, ckpt_manager2)
        models2 = {'g': g2, 'f': f2}

    input_files = {'g': input_files_x, 'f': input_files_y}

    for model, wavs in input_files.items():
        for wav in wavs:
            wave_basename = os.path.basename(wav)
            output_file = os.path.join(SAVE_WAV_PATH, args.model + "-" +
                                       os.path.basename(ckpt_manager.checkpoints[args.epoch - 1]) + "-" +
                                       wave_basename)

            if args.model2:
                ori, pred = predict(wav, models[model], spec_type='harm')
                ori2, pred2 = predict(wav, models2[model], spec_type='perc')
                ori += ori2
                pred += pred2
            else:
                ori, pred = predict(wav, models[model], spec_type=None)

            ori = librosa.util.normalize(ori, norm=np.inf, axis=None)
            pred = librosa.util.normalize(pred, norm=np.inf, axis=None)
            audio_out = np.append(ori, pred)

            make_dirs(os.path.dirname(output_file))

            write_audio(output_file, audio_out, DEFAULT_SAMPLING_RATE)
            print('Prediction saved in', output_file)
