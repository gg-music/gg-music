import librosa
import numpy as np


def to_melspectrogram(songs, n_fft=1024, hop_length=512):
    melspec = lambda x: librosa.feature.melspectrogram(
        x, n_fft=n_fft, hop_length=hop_length)[:, :, np.newaxis]

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


def add_hf(mag, target_shape):
    rec = np.zeros(target_shape)
    rec[0:mag.shape[0], 0:mag.shape[1]] = mag
    return rec


def write_audio(filename, audio, sr=44100):
    librosa.output.write_wav(filename, audio, sr, norm=True)


def join_magnitude_slices(mag_sliced, target_shape, crop_size):
    mag = np.zeros(
        (mag_sliced.shape[1], mag_sliced.shape[0] * mag_sliced.shape[2]))
    for i in range(mag_sliced.shape[0]):
        mag[:, (i) * mag_sliced.shape[2]:(i + 1) *
            mag_sliced.shape[2]] = mag_sliced[i, :, :, 0]
    mag = mag[crop_size[0][0]:target_shape[0] -
              crop_size[0][1], crop_size[1][0]:target_shape[1] -
              crop_size[1][1]]
    return mag


def db_to_amplitude(mag_db, amin=1 / (2**16), normalize=True):
    if (normalize):
        mag_db *= 20 * np.log1p(1 / amin)
    return amin * np.expm1(mag_db / 20)


def inverse_transform(mag, phase, nfft=1024, normalize=True, crop_hf=True):
    window = np.hanning(nfft)
    if (normalize):
        mag = mag * np.sum(np.hanning(nfft)) / 2
    if (crop_hf):
        mag = add_hf(mag, target_shape=(phase.shape[0], mag.shape[1]))
    R = mag * np.exp(1j * phase)
    audio = librosa.istft(R, hop_length=int(nfft / 2), window=window)
    return audio


def write_audio(filename, audio, sr=44100):
    librosa.output.write_wav(filename, audio, sr, norm=True)


def load_audio(filename, sr=22050):
    return librosa.load(filename, sr=sr)[0]


def amplitude_to_db(mag, amin=1 / (2**16), normalize=True):
    mag_db = 20 * np.log1p(mag / amin)
    if (normalize):
        mag_db /= 20 * np.log1p(1 / amin)
    return mag_db


def slice_first_dim(array, slice_size):
    n_sections = int(np.floor(array.shape[1] / slice_size))
    has_last_mag = n_sections * slice_size < array.shape[1]

    last_mag = np.zeros(shape=(1, array.shape[0], slice_size, array.shape[2]))
    last_mag[:, :, :array.shape[1] -
             (n_sections * slice_size), :] = array[:, n_sections *
                                                   int(slice_size):, :]
    if (n_sections > 0):
        array = np.expand_dims(array, axis=0)
        sliced = np.split(array[:, :, 0:n_sections * slice_size, :],
                          n_sections,
                          axis=2)
        sliced = np.concatenate(sliced, axis=0)
        if (has_last_mag):  # Check for reminder
            sliced = np.concatenate([sliced, last_mag], axis=0)
    else:
        sliced = last_mag
    return sliced


def slice_magnitude(mag, slice_size):
    magnitudes = np.stack([mag], axis=2)
    return slice_first_dim(magnitudes, slice_size)
