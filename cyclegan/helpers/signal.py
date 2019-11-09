import librosa
import numpy as np
from ..settings import DEFAULT_SAMPLING_RATE


def load_audio(filename, sr=DEFAULT_SAMPLING_RATE):
    return librosa.load(filename, sr=sr)[0]


def write_audio(filename, audio, sr=DEFAULT_SAMPLING_RATE):
    librosa.output.write_wav(filename, audio, sr, norm=True)


def add_hf(mag, target_shape):
    rec = np.zeros(target_shape)
    rec[0:mag.shape[0], 0:mag.shape[1]] = mag
    return rec


def remove_hf(mag):
    return mag[0:int(mag.shape[0] / 2), :]


def normalize(mag, nfft=1024):
    window = np.hanning(nfft)
    mag = 2 * mag / np.sum(window)
    return mag


def undo_normalize(mag, nfft=1024):
    window = np.hanning(nfft)
    mag = mag * np.sum(window) / 2
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


def to_stft(audio, nfft=1024):
    window = np.hanning(nfft)
    S = librosa.stft(audio, n_fft=nfft, hop_length=int(nfft / 2), window=window)
    return S


def inverse_stft(mag, phase, nfft=1024):
    S = mag * np.exp(1j * phase)
    window = np.hanning(nfft)
    audio = librosa.istft(S, hop_length=int(nfft / 2), window=window)
    return audio


def unet_pad_size(shape, pool_size=2, layers=5):
    pad_size = []
    for length in shape:
        output = length
        for _ in range(layers):
            output = int(np.ceil(output / pool_size))

        padding = output * (pool_size ** layers) - length
        lpad = 0
        rpad = int(padding)
        pad_size.append([lpad, rpad])

    return pad_size


def mag_processing(mag, crop_hf=True, normalized=True, convert_db=True):
    if crop_hf:
        mag = remove_hf(mag)

    if normalized:
        mag = normalize(mag)

    if convert_db:
        mag = amplitude_to_db(mag)
        mag = (mag * 2) - 1

    pad_size = unet_pad_size(mag.shape)

    mag = np.pad(mag, pad_size)
    mag = np.repeat(mag[:, :, np.newaxis], 3, axis=2)
    mag = mag[np.newaxis, :]
    return mag


def mag_inverse(mag, phase, crop_hf=True, normalized=True, convert_db=True):
    mag = mag[0, :, :phase.shape[1], 0]

    if convert_db:
        mag = (mag + 1) / 2
        mag = db_to_amplitude(mag)

    if normalized:
        mag = undo_normalize(mag)

    if crop_hf:
        mag = add_hf(mag, target_shape=[phase.shape[0], mag.shape[1]])

    return mag


def preprocessing_fn(file_path, trim=None, hpss=True, **kwargs):
    signal, sr = librosa.load(file_path, sr=DEFAULT_SAMPLING_RATE)

    if trim:
        trim_length = int(sr * trim)
        signal = signal[:trim_length]

    S = to_stft(signal)

    if hpss:
        S = librosa.decompose.hpss(S)
    else:
        S = [S]

    specs = []
    for D in S:
        mag, phase = librosa.magphase(D)
        mag = mag_processing(mag, **kwargs)
        specs.append([mag, phase])

    return specs


def inverse_fn(mag, phase, trim=True, **kwargs):
    mag = mag_inverse(mag, phase, **kwargs)
    audio_out = inverse_stft(mag, phase)
    if trim:
        # trim 0.1s start/end burst artifact
        audio_out = audio_out[2205:-2205]

    return audio_out


def mel_fq(mag, phase):
    mag = mag_inverse(mag, phase)

    S = librosa.feature.melspectrogram(S=mag, n_mels=256, sr=DEFAULT_SAMPLING_RATE)
    mag, phase = librosa.magphase(S)

    mel_spec = mag_processing(mag, crop_hf=False)
    return mel_spec
