import librosa
import numpy as np

from ..settings import DEFAULT_SAMPLING_RATE


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
    melspec = lambda x: librosa.feature.melspectrogram(
        x, n_fft=n_fft, hop_length=hop_length)[:, :, np.newaxis]

    tsongs = map(melspec, songs)
    return np.array(list(tsongs))


def add_hf(mag, target_shape):
    rec = np.zeros(target_shape)
    rec[0:mag.shape[0], 0:mag.shape[1]] = mag
    return rec


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


def inverse_stft(mag, phase, nfft=1024, normalize=True, crop_hf=True):
    window = np.hanning(nfft)
    if normalize:
        mag = mag * np.sum(np.hanning(nfft)) / 2
    if crop_hf:
        mag = add_hf(mag, target_shape=(phase.shape[0], mag.shape[1]))
    R = mag * np.exp(1j * phase)
    audio = librosa.istft(R, hop_length=int(nfft / 2), window=window)
    return audio


def to_cqt(audio, nfft=1024, normalize=True):
    # A0(27.5Hz) -> B7(3951.066Hz)
    window = np.hanning(int(nfft))
    S = librosa.cqt(audio, n_bins=512, bins_per_octave=12 * 6,
                    hop_length=int(nfft / 2**3),
                    fmin=librosa.note_to_hz('A0'))
    mag, phase = np.abs(S), np.angle(S)
    if normalize:
        mag = 2 * mag / np.sum(window)
    return mag, phase


def inverse_cqt(mag, phase, nfft=1024, normalize=True):
    window = np.hanning(int(nfft))
    if normalize:
        mag = mag * np.sum(window) / 2
    R = mag * np.exp(1j * phase)
    audio = librosa.icqt(R, hop_length=int(nfft / 2**3),
                         bins_per_octave=12 * 6,
                         fmin=librosa.note_to_hz('A0'))
    return audio


def join_magnitude_slices(mag_sliced, target_shape):
    mag = np.zeros((mag_sliced.shape[1], mag_sliced.shape[0] * mag_sliced.shape[2]))
    for i in range(mag_sliced.shape[0]):
        mag[:, (i) * mag_sliced.shape[2]:(i + 1) * mag_sliced.shape[2]] = mag_sliced[i, :, :, 0]
    mag = mag[0:target_shape[0], 0:target_shape[1]]
    return mag


def amplitude_to_db(mag, amin=1 / (2 ** 16), normalize=True):
    mag_db = 20 * np.log1p(mag / amin)
    if normalize:
        mag_db /= 20 * np.log1p(1 / amin)
    return mag_db


def db_to_amplitude(mag_db, amin=1 / (2 ** 16), normalize=True):
    if normalize:
        mag_db *= 20 * np.log1p(1 / amin)
    return amin * np.expm1(mag_db / 20)


def load_audio(filename, sr=DEFAULT_SAMPLING_RATE):
    return librosa.load(filename, sr=sr)[0]


def write_audio(filename, audio, sr=DEFAULT_SAMPLING_RATE):
    librosa.output.write_wav(filename, audio, sr, norm=True)


def unet_pad_size(shape, pool_size=2, layers=5):
    pad_size = []
    for length in shape:
        output = length
        for _ in range(layers):
            output = int(np.ceil(output / pool_size))

        padding = output * (pool_size ** layers) - length
        lpad = int(np.ceil(padding / 2))
        rpad = int(np.floor(padding / 2))
        pad_size.append([lpad, rpad])

    return pad_size


def crop(image, crop_size):
    upad = crop_size[0][0]
    dpad = crop_size[0][1]
    lpad = crop_size[1][0]
    rpad = crop_size[1][1]
    image = image[:, upad:image.shape[1] - dpad, lpad:image.shape[2] - rpad, :]
    return image


def preprocessing_fn(file_path, spec_format, chan=3,
                     trim=None, split=None, convert_db=True):
    spec_types = {0: to_stft, 1: to_cqt}
    signal, sr = librosa.load(file_path, sr=DEFAULT_SAMPLING_RATE)

    if trim:
        trim_length = int(sr * trim)
        signal = signal[:trim_length]

    if split:
        signal = splitsongs(signal, window=split)

    mag, phase = spec_types[spec_format](signal)

    if np.max(mag) == 0:
        raise ValueError

    if convert_db:
        mag = amplitude_to_db(mag)
        mag = (mag * 2) - 1

    pad_size = unet_pad_size(mag.shape)

    mag = np.pad(mag, pad_size)
    mag = np.repeat(mag[:, :, np.newaxis], chan, axis=2)

    return mag, phase


def inverse_fn(mag, phase, spec_format, convert_db=True, trim=True):
    inverse_type = {0: inverse_stft, 1: inverse_cqt}

    if convert_db:
        mag = (mag + 1) / 2
        mag = db_to_amplitude(mag)

    mag = join_magnitude_slices(mag, target_shape=phase.shape)
    audio_out = inverse_type[spec_format](mag, phase)
    if trim:
        # trim start/end burst artifact
        audio_out = audio_out[2000:-2000]

    return audio_out
