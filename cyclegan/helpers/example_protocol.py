import os

import numpy as np
import tensorflow as tf
import librosa
from .signal import mag_processing


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
        'height': _int64_feature(np_array.shape[0]),
        'width': _int64_feature(np_array.shape[1]),
        'title': _bytes_feature(title.encode('utf-8')),
        'spec': _bytes_feature(np_array.tostring())
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def extract_example(example):
    feature = {
        'height': example.features.feature['height'].int64_list.value[0],
        'width': example.features.feature['width'].int64_list.value[0],
        'title': example.features.feature['title'].bytes_list.value[0].decode('utf-8')
    }

    spec = example.features.feature['spec'].bytes_list.value[0]
    spec = np.fromstring(spec, dtype=np.complex128)
    spec = spec.astype(np.complex64)
    spec = spec.reshape((feature['height'], feature['width']))

    harm_spec, perc_spec = librosa.decompose.hpss(spec)
    specs = {'ori': spec, 'harm': harm_spec, 'perc': perc_spec}

    for k, v in specs.items():
        mag, phase = librosa.magphase(v)
        mag = mag_processing(mag)

        feature[f'{k}_spec'] = v
        feature[f'{k}_mag'] = mag
        feature[f'{k}_phase'] = phase

    return feature
