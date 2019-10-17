import librosa
import numpy as np


def to_melspectrogram(songs, n_fft=1024, hop_length=512):
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft=n_fft,
                                                       hop_length=hop_length)[:, :, np.newaxis]

    tsongs = map(melspec, songs)
    return np.array(list(tsongs))


def remove_hf(mag):
    return mag[0:int(mag.shape[0] / 2), :]


def to_stft(audio, nfft=1024, normalize=True, crop_hf=True):
    window = np.hanning(nfft)
    S = librosa.stft(audio, n_fft=nfft, hop_length=int(nfft / 2), window=window)
    mag, phase = np.abs(S), np.angle(S)
    if crop_hf:
        mag = remove_hf(mag)
    if normalize:
        mag = 2 * mag / np.sum(window)
    return mag, phase


def splitsongs(X, window=0.1, overlap=0.5):
    temp_X = []

    xshape = X.shape[0]
    chunk = int(xshape * window)
    offset = int(chunk * (1. - overlap))

    spsong = [X[i:i + chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)

    return np.array(temp_X)
