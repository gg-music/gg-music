from multiprocessing import Pool, cpu_count
import audioread
import numpy as np
import os

from cyclegan.helpers.signal import preprocessing_fn
from .utils import make_dirs
from .plot import plot_heat_map


def batch(iterable, n=1):
    iter_len = len(iterable)
    for ndx in range(0, iter_len, n):
        yield iterable[ndx:min(ndx + n, iter_len)]


def processing(file_list, par, batch_size=10):

    pool = Pool(processes=cpu_count(), maxtasksperchild=1)

    for _ in pool.imap_unordered(par, batch(file_list, batch_size)):
        pass

    pool.close()
    pool.join()


def batch_plot(batch_file_path, output_dir, **kwargs):
    for file_path in batch_file_path:
        title = file_path.split('/')[-1].split('.')[0]
        save_dir = os.path.join(output_dir,
                                os.path.join(file_path.split('/')[-2]))
        if 'npy' in file_path:
            # print(np.load(file_path))
            plot_heat_map(np.load(file_path), title, save_dir)
        else:
            pass
        # if '.log' in file_path:
        #     pass
        # else:
        #     """
        #         npy
        #     """
        #     plot_heat_map()
        # pass
        # batch_specs = []
        # try:
        #     specs, _ = preprocessing_fn(file_path, spec_format, **kwargs)
        # except ValueError:
        #     os.remove(file_path)
        #     print("\nremove zero file: " + file_path + "\n")
        #     continue
        # except audioread.exceptions.NoBackendError as err:
        #     print("\n", err, file_path, "\n")
        #     continue

        # batch_specs.append(specs)
        # batch_specs = np.array(batch_specs)

        # file_name = os.path.basename(file_path).split('.')[-2]
        # category = os.path.dirname(file_path).split('/')[-1]
        # category_dir = os.path.join(output_dir, category)

        # make_dirs(category_dir)

        # if to_tfrecord:
        #     output2tfrecord(category_dir, file_name, batch_specs)
        # else:
        #     output2raw(category_dir, file_name, batch_specs)


def batch_processing(batch_file_path,
                     output_dir,
                     spec_format,
                     to_tfrecord=False,
                     **kwargs):
    for file_path in batch_file_path:
        batch_specs = []
        try:
            specs, _ = preprocessing_fn(file_path, spec_format, **kwargs)
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

        if to_tfrecord:
            output2tfrecord(category_dir, file_name, batch_specs)
        else:
            output2raw(category_dir, file_name, batch_specs)


def output2raw(category_dir, file_name, batch_specs):
    save_file = os.path.join(category_dir, '{}.npy'.format(file_name))
    np.save(save_file, batch_specs)

    print(f'{save_file}')


def output2tfrecord(category_dir, file_name, batch_specs):
    import tensorflow as tf
    from .example_protocol import np_array_to_example

    save_file = os.path.join(category_dir, '{}.tfrecords'.format(file_name))
    with tf.device('/cpu:0'):
        with tf.io.TFRecordWriter(save_file) as writer:
            tf_example = np_array_to_example(batch_specs, save_file)
            writer.write(tf_example)

    print(f'{save_file}')
