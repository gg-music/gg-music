import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import librosa
import tensorflow as tf
from cyclegan.segmentation_models.nestnet import Nestnet as Generator
from cyclegan.model.vgg_model import vgg16_model as Discriminator
from cyclegan.helpers.signal import preprocessing_fn, inverse_fn, write_audio, mel_fq
from cyclegan.helpers.plot import plot_heat_map
from cyclegan.settings import DEFAULT_SAMPLING_RATE


TEST_DIR = '/home/gtzan/ssd/test'
inp = '/home/gtzan/data/gan_preprocessing/wav/sax/sax-0003.wav'

[spec] = preprocessing_fn(inp, hpss=False)
harm_spec, perc_spec = preprocessing_fn(inp)

plot_heat_map(spec[0], title='real_x', save_dir=TEST_DIR)

audio_out = inverse_fn(spec[0], spec[1])
output_filename = os.path.join(TEST_DIR, f'clear_{os.path.basename(inp)}')
write_audio(output_filename, audio_out, DEFAULT_SAMPLING_RATE)

plot_heat_map(harm_spec[0], title='harmonic_x', save_dir=TEST_DIR)
audio_out = inverse_fn(harm_spec[0], harm_spec[1])
output_filename = os.path.join(TEST_DIR, f'harmonic_{os.path.basename(inp)}')
write_audio(output_filename, audio_out, DEFAULT_SAMPLING_RATE)

plot_heat_map(perc_spec[0], title='percussion_x', save_dir=TEST_DIR)
audio_out = inverse_fn(perc_spec[0], perc_spec[1])
output_filename = os.path.join(TEST_DIR, f'percussion_{os.path.basename(inp)}')
write_audio(output_filename, audio_out, DEFAULT_SAMPLING_RATE)


generator_g = Generator(backbone_name='vgg16',
                        input_shape=(None, None, 3),
                        decoder_filters=(256, 128, 64, 32, 16),
                        classes=3,
                        activation='tanh')

discriminator_x = Discriminator(norm_type='instancenorm', target=False)

fake_x = generator_g(spec[0])
plot_heat_map(fake_x, title='fake_x', save_dir=TEST_DIR)

plot_heat_map(mel_fq(spec[0], spec[1]), title='mel_real_x', save_dir=TEST_DIR)

disc_real_x = discriminator_x(mel_fq(spec[0], spec[1]))
plot_heat_map(disc_real_x, title='disc_real_x', save_dir=TEST_DIR)

disc_fake_x = discriminator_x(mel_fq(fake_x, spec[1]))
plot_heat_map(disc_fake_x, title='disc_fake_x', save_dir=TEST_DIR)
