from multiprocessing import Pool, cpu_count
import audioread
import numpy as np
import os

from .signal import preprocessing_fn
from .utils import make_dirs
from .plot import plot_heat_map, plot_epoch_loss_by_log


def batch(iterable, n=1):
    iter_len = len(iterable)
    for ndx in range(0, iter_len, n):
        yield iterable[ndx:min(ndx + n, iter_len)]


def processing(file_list, par, batch_size=100):
    pool = Pool(processes=cpu_count(), maxtasksperchild=1)

    for _ in pool.imap_unordered(par, batch(file_list, batch_size)):
        pass

    pool.close()
    pool.join()


def batch_plot(batch_file_path, output_dir):
    for file_path in batch_file_path:

        title = file_path.split('/')[-1][:-4]

        save_dir = os.path.join(output_dir,
                                os.path.join(file_path.split('/')[-2]))
        if 'npy' in file_path:
            plot_heat_map(np.load(file_path), title, save_dir)
        else:
            plot_epoch_loss_by_log(
                np.genfromtxt(file_path, delimiter=',')[:-1], save_dir, title)
        print(title)


def batch_processing(batch_file_path,
                     output_dir,
                     **kwargs):
    for file_path in batch_file_path:
        try:
            spec = preprocessing_fn(file_path, **kwargs)
        except ValueError:
            os.remove(file_path)
            print("\nremove zero file: " + file_path + "\n")
            continue
        except audioread.exceptions.NoBackendError as err:
            print("\n", err, file_path, "\n")
            continue

        spec = np.array(spec)

        file_name = os.path.basename(file_path).split('.')[-2]
        category = os.path.dirname(file_path).split('/')[-1]
        category = f'{category}'
        category_dir = os.path.join(output_dir, category)

        make_dirs(category_dir)
        output2tfrecord(category_dir, file_name, spec)


def output2tfrecord(category_dir, file_name, spec):
    import tensorflow as tf
    from .example_protocol import np_array_to_example

    save_file = os.path.join(category_dir, '{}.tfrecord'.format(file_name))
    with tf.device('/cpu:0'):
        with tf.io.TFRecordWriter(save_file) as writer:
            tf_example = np_array_to_example(spec, save_file)
            writer.write(tf_example)

    print(save_file)
