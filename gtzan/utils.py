import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import audioread
import librosa
import numpy as np
from scipy import stats

from gtzan.signal import splitsongs


def batch(iterable, n=1):
    iter_len = len(iterable)
    for ndx in range(0, iter_len, n):
        yield iterable[ndx:min(ndx + n, iter_len)]


def parallel_preprocessing(song_list, output_dir,
                           spec_format=None, category=None,
                           batch_size=10, **kwargs):
    par = partial(preprocessing,
                  category=category,
                  output_dir=output_dir,
                  spec_format=spec_format,
                  **kwargs)

    pool = Pool(processes=cpu_count(), maxtasksperchild=1)

    for _ in pool.imap_unordered(par, batch(song_list, batch_size)):
        pass

    pool.close()
    pool.join()


def preprocessing(batch_file_path, output_dir,
                  spec_format, category,
                  trim=None, split=None):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir, mode=0o777)

    for file_path in batch_file_path:
        arr_specs = []
        try:
            signal, sr = librosa.load(file_path)

            if trim:
                trim_length = sr * trim
                signal = signal[:trim_length]

            if split:
                signal = splitsongs(signal, window=split)

            specs = spec_format(signal)

        except ValueError:
            continue
        except audioread.exceptions.NoBackendError:
            continue

        arr_specs.extend(specs)

        file_name = os.path.basename(file_path).split('.')[-2]

        if not category:
            category = os.path.dirname(file_path).split('/')[-1]

        category_dir = os.path.join(output_dir, category)

        if not os.path.isdir(category_dir):
            os.mkdir(category_dir, mode=0o777)
        save_file = os.path.join(category_dir, '{}.npy'.format(file_name))

        np.save(save_file, arr_specs)
        print('{}'.format(save_file))


def get_file_list(src_dir, catalog_offset=0):
    input_path = []
    category = []
    for dir_path, subdir, filenames in os.walk(src_dir):
        for f in filenames:
            input_path.append(os.path.join(dir_path, f))

            if catalog_offset:
                dir_array = dir_path.split('/')
                cat = dir_array[catalog_offset]
                reverse_mapping = load_mapping(reverse=True)
                category.append(reverse_mapping[cat])

    if catalog_offset:
        return input_path, category
    else:
        return input_path


def load_mapping(reverse=False):
    file = '/home/gtzan/category_label_mapping.json'
    with open(file, 'r') as f:
        mapping = json.load(f)

    if reverse:
        reverse_mapping = {v: k for k, v in mapping.items()}
        return reverse_mapping
    else:
        return mapping


def pred_to_y(pred, n_song, split_per_song):
    total = n_song * split_per_song
    max_prob = np.argmax(pred, axis=1)
    group_by_song = np.array(max_prob[:total]).reshape((n_song, split_per_song))
    song_mode = stats.mode(group_by_song, axis=1)
    y_pred = np.array(song_mode[0]).reshape(n_song, )

    return y_pred
