import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import audioread
import librosa
import numpy as np


def batch(iterable, n=1):
    iter_len = len(iterable)
    for ndx in range(0, iter_len, n):
        yield iterable[ndx:min(ndx + n, iter_len)]


def splitsongs(X, window=0.1, overlap=0.5):
    temp_X = []

    xshape = X.shape[0]
    chunk = int(xshape * window)
    offset = int(chunk * (1. - overlap))

    spsong = [X[i:i + chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)

    return np.array(temp_X)


def to_melspectrogram(songs, n_fft=1024, hop_length=512):
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft=n_fft,
                                                       hop_length=hop_length)[:, :, np.newaxis]

    tsongs = map(melspec, songs)
    return np.array(list(tsongs))


def parallel_preprocessing(song_list, output_dir, category=None, batch_size=10):
    par = partial(preprocessing, category=category, output_dir=output_dir, spec_format=to_melspectrogram)
    pool = Pool(processes=cpu_count(), maxtasksperchild=1)

    for _ in pool.imap_unordered(par, batch(song_list, batch_size)):
        pass

    pool.close()
    pool.join()


def preprocessing(batch_file_path, output_dir, spec_format, category):
    song_samples = 660000
    for file_path in batch_file_path:
        arr_specs = []
        try:
            signal, sr = librosa.load(file_path)
            signal = signal[:song_samples]

            signals = splitsongs(signal)
            specs = spec_format(signals)

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
                reverse_mapping = load_mapping(reverse=True)
                dir_array = dir_path.split('/')
                cat = dir_array[catalog_offset]
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
