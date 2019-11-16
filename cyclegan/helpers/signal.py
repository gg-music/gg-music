import librosa
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
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


def mag_phase_to_S(mag, phase):
    S = mag * np.exp(1j * phase)
    return S


def amplitude_to_db(mag, amin=1 / (2 ** 16), normalize=True):
    mag_db = 20 * np.log1p(mag / amin)
    if normalize:
        mag_db /= 20 * np.log1p(1 / amin)
    return mag_db


def db_to_amplitude(mag_db, amin=1 / (2 ** 16), normalize=True):
    if normalize:
        mag_db *= 20 * np.log1p(1 / amin)
    return amin * tf.math.expm1(mag_db / 20)


def to_stft(audio, nfft=1024):
    window = np.hanning(nfft)
    S = librosa.stft(audio, n_fft=nfft, hop_length=int(nfft / 2), window=window)
    return S


def inverse_stft(mag, phase, nfft=1024):
    S = mag_phase_to_S(mag, phase)
    window = np.hanning(nfft)
    audio = librosa.istft(S, hop_length=int(nfft / 2), window=window)
    return audio


def to_cqt(audio, nfft=1024):
    # A0(27.5Hz) -> B7(3951.066Hz)
    window = np.hanning(int(nfft))
    S = librosa.cqt(audio, n_bins=512, bins_per_octave=12 * 6,
                    hop_length=int(nfft / 2),
                    fmin=librosa.note_to_hz('A0'))
    mag, phase = librosa.magphase(S)
    return mag, phase


def inverse_cqt(mag, phase, nfft=1024):
    S = mag_phase_to_S(mag, phase)
    audio = librosa.icqt(S, hop_length=int(nfft / 2 ** 3),
                         bins_per_octave=12 * 6,
                         fmin=librosa.note_to_hz('A0'))
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
    mag = np.repeat(mag[:, :, np.newaxis], 3, axis=-1)
    mag = mag[np.newaxis, :]
    return mag


def mag_inverse(mag, target_shape, crop_hf=True, normalized=True, convert_db=True):
    mag = mag[0, :, :target_shape[1], 0]

    if convert_db:
        mag = (mag + 1) / 2
        mag = db_to_amplitude(mag)

    if normalized:
        mag = undo_normalize(mag)

    if crop_hf:
        mag = add_hf(mag, target_shape=target_shape)

    return mag


def inverse_fn(mag, phase, trim=True, **kwargs):
    mag = mag_inverse(mag, phase.shape, **kwargs)
    audio_out = inverse_stft(mag, phase)
    if trim:
        # trim 0.1s start/end burst artifact
        audio_out = audio_out[2205:-2205]

    return audio_out


def log_fq(spec, nfft=1024, sr=DEFAULT_SAMPLING_RATE):
    y_ref = tf.transpose(spec[0, :, :, 0])

    nyquest = sr / 2
    low_fq = nyquest / (1 + nfft / 2)
    high_fq = y_ref.shape[1] * nyquest / (1 + nfft / 2)

    exp_fq = tf.exp(tf.linspace(tf.math.log(low_fq), tf.math.log(high_fq), y_ref.shape[1]))

    y = tfp.math.batch_interp_regular_1d_grid(
        exp_fq,
        x_ref_min=low_fq,
        x_ref_max=high_fq,
        y_ref=y_ref,
        axis=1)
    y = tf.transpose(y)

    log_spec = tf.stack([y for i in range(3)], axis=-1)
    log_spec = log_spec[tf.newaxis, :, :, :]

    return log_spec


def mel_spec(mag):
    shape = (513, 255)
    mag = mag_inverse(mag, shape)
    mag = librosa.feature.melspectrogram(S=mag, n_mels=256, sr=DEFAULT_SAMPLING_RATE)
    mag = mag_processing(mag, crop_hf=False)
    return mag


def preprocessing_fn(file_path, spec_type=None, trim=None, **kwargs):
    signal, sr = librosa.load(file_path, sr=DEFAULT_SAMPLING_RATE)

    if trim:
        trim_length = int(sr * trim)
        signal = signal[:trim_length]
    spec = to_stft(signal)

    if spec_type:
        specs = {}
        specs['harm'], specs['perc'] = librosa.decompose.hpss(spec)
        spec = specs[spec_type]

    mag, phase = librosa.magphase(spec)
    mag = mag_processing(mag, **kwargs)

    return mag, phase


def predict(inp, model, spec_type=None):
    mag, phase = preprocessing_fn(inp, spec_type)
    ori = inverse_fn(mag, phase)

    mag = model.predict(mag)
    pred = inverse_fn(mag, phase)

    audio_out = np.append(ori, pred)
    return audio_out
