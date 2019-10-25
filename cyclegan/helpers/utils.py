import os
from functools import partial
from multiprocessing import Pool, cpu_count

import audioread
import librosa
import numpy as np
import tensorflow as tf

from .example_protocol import np_array_to_example
from .signal import splitsongs, amplitude_to_db
from ..settings import PAD_SIZE, DEFAULT_SAMPLING_RATE


def get_file_list(src_dir):
    input_path = []
    for dir_path, subdir, filenames in os.walk(src_dir):
        for f in filenames:
            input_path.append(os.path.join(dir_path, f))

    return input_path


def make_dirs(path):
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o777)


def batch(iterable, n=1):
    iter_len = len(iterable)
    for ndx in range(0, iter_len, n):
        yield iterable[ndx:min(ndx + n, iter_len)]


def parallel_preprocessing(song_list,
                           output_dir,
                           spec_format=None,
                           batch_size=10,
                           **kwargs):

    par = partial(batch_preprocessing,
                  output_dir=output_dir,
                  spec_format=spec_format,
                  **kwargs)

    pool = Pool(processes=cpu_count(), maxtasksperchild=1)

    for _ in pool.imap_unordered(par, batch(song_list, batch_size)):
        pass

    pool.close()
    pool.join()


def batch_preprocessing(batch_file_path,
                        output_dir,
                        spec_format,
                        **kwargs):
    for file_path in batch_file_path:
        batch_specs = []
        try:
            specs, _ = preprocessing_fn(file_path,
                                     spec_format,
                                     **kwargs)

        except ValueError:
            os.remove(file_path)
            print("\nremove zero file: " + file_path + "\n")
            continue
        except audioread.exceptions.NoBackendError as err:
            print("\n", err, file_path, "\n")
            continue

        batch_specs.append(specs)
        batch_specs = np.array(batch_specs)

        file_name = os.path.basename(file_path).split('.')[-2]
        category = os.path.dirname(file_path).split('/')[-1]
        category_dir = os.path.join(output_dir, category)

        make_dirs(category_dir)
        save_file = os.path.join(category_dir, '{}.tfrecords'.format(file_name))
        with tf.device('/cpu:0'):
            with tf.io.TFRecordWriter(save_file) as writer:
                tf_example = np_array_to_example(batch_specs, save_file)
                writer.write(tf_example)

        print('{}'.format(save_file))


def preprocessing_fn(file_path,
                     spec_format,
                     trim=None,
                     split=None,
                     convert_db=True,
                     pad_size=PAD_SIZE):
    signal, sr = librosa.load(file_path, sr=DEFAULT_SAMPLING_RATE)
    if trim:
        trim_length = sr * trim
        signal = signal[:trim_length]

    if split:
        signal = splitsongs(signal, window=split)

    mag, phase = spec_format(signal)

    if np.max(mag) == 0:
        raise ValueError

    if convert_db:
        mag = amplitude_to_db(mag)
        mag = (mag * 2) - 1

    if pad_size:
        mag = np.pad(mag, pad_size)
        mag = np.repeat(mag[:, :, np.newaxis], 3, axis=2)

    return mag, phase


