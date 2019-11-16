import tensorflow as tf
from ..helpers.utils import get_file_list
from ..helpers.example_protocol import extract_example
from ..helpers.plot import plot_heat_map
from ..settings import MUSIC_NPY_PATH

x_list = get_file_list(MUSIC_NPY_PATH['sax'])

dataset = tf.data.TFRecordDataset(x_list)

for example in dataset.take(1):
    example = tf.train.Example.FromString(example.numpy())
    feature = extract_example(example)
    plot_heat_map(feature['data'], feature['title'], '.')
