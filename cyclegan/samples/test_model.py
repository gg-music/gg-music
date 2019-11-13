import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import librosa
from cyclegan.segmentation_models.nestnet import Nestnet as Generator
from cyclegan.model.vgg_model import vgg16_model as Discriminator

from cyclegan.helpers.signal import preprocessing_fn, inverse_fn, write_audio, mag_phase_to_S, mel_spec
from cyclegan.helpers.signal import mag_processing, mag_inverse, to_cqt, load_audio, log_fq
from cyclegan.helpers.plot import plot_heat_map
from cyclegan.settings import DEFAULT_SAMPLING_RATE
import tensorflow as tf

TEST_DIR = '/home/gtzan/ssd/test'
inp = '/home/gtzan/data/gan_preprocessing/wav/piano/piano1-0331.wav'

# CQT
audio = load_audio(inp, sr=DEFAULT_SAMPLING_RATE)
mag, phase = to_cqt(audio)
mag = mag_processing(mag, crop_hf=False)
plot_heat_map(mag, title='piano_cqt')

# STFT
mag, phase = preprocessing_fn(inp)
print(mag.shape)
plot_heat_map(mel_spec(mag), title='piano_mel')

audio_out = inverse_fn(mag, phase)
output_filename = os.path.join(TEST_DIR, f'clear_{os.path.basename(inp)}')
write_audio(output_filename, audio_out, DEFAULT_SAMPLING_RATE)

mag = mag_inverse(mag, phase.shape)
S = mag_phase_to_S(mag, phase)

harm, perc = librosa.decompose.hpss(S)
harm_mag, harm_phase = librosa.magphase(harm)
perc_mag, perc_phase = librosa.magphase(perc)
harm_mag = mag_processing(harm_mag)
perc_mag = mag_processing(perc_mag)

plot_heat_map(harm_mag, title='harmonic')
audio_out = inverse_fn(harm_mag, phase)
output_filename = os.path.join(TEST_DIR, f'harmonic_{os.path.basename(inp)}')
write_audio(output_filename, audio_out, DEFAULT_SAMPLING_RATE)

plot_heat_map(perc_mag, title='percussion')
audio_out = inverse_fn(perc_mag, phase)
output_filename = os.path.join(TEST_DIR, f'percussion_{os.path.basename(inp)}')
write_audio(output_filename, audio_out, DEFAULT_SAMPLING_RATE)


generator_g = Generator(backbone_name='vgg16',
                        input_shape=(None, None, 3),
                        decoder_filters=(256, 128, 64, 32, 16),
                        classes=3,
                        activation='tanh')

discriminator_x = Discriminator(norm_type='instancenorm', target=False)

mag = mag_processing(mag)

fake_x = generator_g(mag)
plot_heat_map(fake_x, title='fake_x')

plot_heat_map(mel_spec(mag), title='mel_real_x')

disc_real_x = discriminator_x(mag)
plot_heat_map(disc_real_x, title='disc_real_x')

disc_fake_x = discriminator_x(mel_spec(fake_x))
plot_heat_map(disc_fake_x, title='disc_fake_x')
