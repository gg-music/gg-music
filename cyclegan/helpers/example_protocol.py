import os

import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def np_array_to_example(harm, prec, file):
    title = os.path.basename(file).split('.')[-2]
    feature = {
        'title': _bytes_feature(title.encode('utf-8')),

        'harm': _bytes_feature(harm[0].astype(dtype=np.float32).tostring()),
        'harm_height': _int64_feature(harm[0].shape[0]),
        'harm_width': _int64_feature(harm[0].shape[1]),
        'harm_chan': _int64_feature(harm[0].shape[2]),

        'prec': _bytes_feature(prec[0].astype(dtype=np.float32).tostring()),
        'prec_height': _int64_feature(prec[0].shape[0]),
        'prec_width': _int64_feature(prec[0].shape[1]),
        'prec_chan': _int64_feature(prec[0].shape[2]),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def extract_example(example):
    feature = {
        'title': example.features.feature['title'].bytes_list.value[0].decode('utf-8'),

        'harm_height': example.features.feature['harm_height'].int64_list.value[0],
        'harm_width': example.features.feature['harm_width'].int64_list.value[0],
        'harm_chan': example.features.feature['harm_chan'].int64_list.value[0],

        'prec_height': example.features.feature['prec_height'].int64_list.value[0],
        'prec_width': example.features.feature['prec_width'].int64_list.value[0],
        'prec_chan': example.features.feature['prec_chan'].int64_list.value[0],
    }
    for m in ['harm', 'prec']:
        mag = example.features.feature[m].bytes_list.value[0]
        mag = np.fromstring(mag, dtype=np.float32)
        mag = mag.reshape((1, feature[f'{m}_height'], feature[f'{m}_width'], feature[f'{m}_chan']))
        feature[m] = mag

    return feature
