import os
import librosa
import numpy as np
import audioread
from functools import partial
from multiprocessing import Pool, cpu_count


def batch(iterable, n=1):
    iter_len = len(iterable)
    for ndx in range(0, iter_len, n):
        yield iterable[ndx:min(ndx + n, iter_len)]


def splitsongs(X, window=0.1, overlap=0.5):
    # Empty lists to hold our results
    temp_X = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape * window)
    offset = int(chunk * (1. - overlap))

    # Split the song and create new ones on windows
    spsong = [X[i:i + chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)

    return np.array(temp_X)


def to_melspectrogram(songs, n_fft=1024, hop_length=512):
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft=n_fft,
                                                       hop_length=hop_length)[:, :, np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    return np.array(list(tsongs))


def parallel_preprocessing(src_csv, output_dir, batch_size):
    song_list, category = read_csv(src_csv)

    par = partial(preprocessing, output_dir=output_dir, spec_format=to_melspectrogram)
    pool = Pool(processes=cpu_count(), maxtasksperchild=1)

    for _ in pool.imap_unordered(par, batch(song_list, batch_size)):
        pass

    pool.close()
    pool.join()


def preprocessing(batch_file_path, output_dir, spec_format):
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

        file_name = os.path.basename(file_path).split('.')[0]
        save_file = os.path.join(output_dir, '{}.npy'.format(file_name))
        np.save(save_file, arr_specs)
        print('{}'.format(save_file))


def read_csv(src_csv):
    csv = []
    with open(src_csv, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            content = line.strip().split(',')
            csv.append(content)

    return csv

