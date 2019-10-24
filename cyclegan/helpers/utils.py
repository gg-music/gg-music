import os
from functools import partial
from multiprocessing import Pool, cpu_count

import audioread
import librosa
import numpy as np
import tensorflow as tf
from .signal import splitsongs, amplitude_to_db
from ..settings import PAD_SIZE


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
    signal, sr = librosa.load(file_path)
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


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def np_array_to_example(np_array, file):
    title = os.path.basename(file).split('.')[-2]
    feature = {
        'height': _int64_feature(np_array.shape[1]),
        'width': _int64_feature(np_array.shape[2]),
        'depth': _int64_feature(np_array.shape[3]),
        'title': _bytes_feature(title.encode('utf-8')),
        'data': _bytes_feature(np_array.astype(dtype=np.float32).tostring())
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def extract_example(example):
    feature = {
        'height': example.features.feature['height'].int64_list.value[0],
        'width': example.features.feature['width'].int64_list.value[0],
        'depth': example.features.feature['depth'].int64_list.value[0],
        'title': example.features.feature['title'].bytes_list.value[0].decode('utf-8')
    }

    data = example.features.feature['data'].bytes_list.value[0]
    data = np.fromstring(data, dtype=np.float32)
    data = data.reshape((1, feature['height'], feature['width'], feature['depth']))
    feature['data'] = data

    return feature
