import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import librosa
import tensorflow as tf
from cyclegan.segmentation_models.nestnet import Nestnet as Generator
from cyclegan.model.vgg_model import vgg16_model as Discriminator
from cyclegan.helpers.signal import preprocessing_fn, inverse_fn, write_audio, mel_spec, mag_processing
from cyclegan.helpers.plot import plot_heat_map
from cyclegan.settings import DEFAULT_SAMPLING_RATE

TEST_DIR = '/home/gtzan/ssd/test'
inp = '/home/gtzan/data/gan_preprocessing/wav/sax/sax-0003.wav'
test = {}

spec = preprocessing_fn(inp)
test['spec'] = {'ori': spec}
test['mag'] = {}
test['phase'] = {}
test['spec']['harm'], test['spec']['perc'] = librosa.decompose.hpss(spec)

for k, v in test['spec'].items():
    mag, phase = librosa.magphase(spec)
    mag = mag_processing(mag)

    test['mag'][k] = mag
    test['phase'][k] = phase


for k, _ in test['spec'].items():
    plot_heat_map(test['mag'][k], title=f'{k}_x', save_dir=TEST_DIR)
    audio_out = inverse_fn(test['mag'][k], test['phase'][k])
    output_filename = os.path.join(TEST_DIR, f'{k}_{os.path.basename(inp)}')
    write_audio(output_filename, audio_out, DEFAULT_SAMPLING_RATE)


generator_g = Generator(backbone_name='vgg16',
                        input_shape=(None, None, 3),
                        decoder_filters=(256, 128, 64, 32, 16),
                        classes=3,
                        activation='tanh')

discriminator_x = Discriminator(norm_type='instancenorm', target=False)

fake_x = generator_g(test['mag']['ori'])
plot_heat_map(fake_x, title='fake_x', save_dir=TEST_DIR)
mel = mel_spec(test['mag']['ori'], test['phase']['ori'])
plot_heat_map(mel, title='mel_real_x', save_dir=TEST_DIR)

disc_real_x = discriminator_x(mel_spec(test['mag']['ori'], test['phase']['ori']))
plot_heat_map(disc_real_x, title='disc_real_x', save_dir=TEST_DIR)

disc_fake_x = discriminator_x(mel_spec(test['mag']['ori'], test['phase']['ori']))
plot_heat_map(disc_fake_x, title='disc_fake_x', save_dir=TEST_DIR)
