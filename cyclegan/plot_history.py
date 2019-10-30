import os
import argparse
from functools import partial
from .helpers.utils import get_file_list, make_dirs
from .helpers.parallel import batch_plot, processing
from .settings import MODEL_ROOT_PATH
from .helpers.plot import init_plotter
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, help='model name', type=str)

    ap.add_argument('-pd',
                    '--plot_dst',
                    required=True,
                    help='plot npy or log',
                    type=str)

    ap.add_argument('--batch_size',
                    required=False,
                    default=10,
                    help='files per batch',
                    type=int)

    args = ap.parse_args()

    if args.plot_dst not in ('npy', 'log'):
        raise ValueError('invalid plot_dst')

    root_path = os.path.join(MODEL_ROOT_PATH, os.path.basename(args.model))
    if not os.path.isdir(root_path):
        raise FileNotFoundError('Invalid src path')

    src_folder = 'logs' if args.plot_dst == 'log' else 'npy'
    src_path = os.path.join(root_path, src_folder)
    image_path = os.path.join(root_path, 'images')

    file_list = get_file_list(src_path)

    make_dirs(image_path)

    par = partial(batch_plot, output_dir=image_path)

    processing(file_list, par, args.batch_size)

# def parallel_preprocessing(song_list,
#                            output_dir,
#                            spec_format=None,
#                            batch_size=10,
#                            **kwargs):

#     par = partial(batch_processing,
#                   output_dir=output_dir,
#                   spec_format=spec_format,
#                   **kwargs)

#     pool = Pool(processes=cpu_count(), maxtasksperchild=1)

#     for _ in pool.imap_unordered(par, batch(song_list, batch_size)):
#         pass

#     pool.close()
#     pool.join()

# def batch_processing(batch_file_path, output_dir, spec_format, **kwargs):
#     for file_path in batch_file_path:
#         batch_specs = []
#         try:
#             specs, _ = preprocessing_fn(file_path, spec_format, **kwargs)

#         except ValueError:
#             os.remove(file_path)
#             print("\nremove zero file: " + file_path + "\n")
#             continue
#         except audioread.exceptions.NoBackendError as err:
#             print("\n", err, file_path, "\n")
#             continue

#         batch_specs.append(specs)
#         batch_specs = np.array(batch_specs)

#         file_name = os.path.basename(file_path).split('.')[-2]
#         category = os.path.dirname(file_path).split('/')[-1]
#         category_dir = os.path.join(output_dir, category)

#         make_dirs(category_dir)
#         save_file = os.path.join(category_dir, '{}.tfrecords'.format(file_name))
#         with tf.device('/cpu:0'):
#             with tf.io.TFRecordWriter(save_file) as writer:
#                 tf_example = np_array_to_example(batch_specs, save_file)
#                 writer.write(tf_example)

#         print('{}'.format(save_file))
