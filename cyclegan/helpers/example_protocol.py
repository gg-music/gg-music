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
