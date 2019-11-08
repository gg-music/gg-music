import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cyclegan.segmentation_models.nestnet import Nestnet as Generator
from cyclegan.model.vgg_model import vgg16_model as Discriminator
from cyclegan.helpers.signal import preprocessing_fn, inverse_fn, write_audio, mel_fq
from cyclegan.helpers.plot import plot_heat_map
from cyclegan.settings import DEFAULT_SAMPLING_RATE
import tensorflow as tf

TEST_DIR = '/home/gtzan/ssd/test'
inp = '/home/gtzan/data/gan_preprocessing/wav/string1/string1-0003.wav'

mag, phase = preprocessing_fn(inp, spec_format=0)
mag = mag[tf.newaxis, :]
print(mag.shape)
plot_heat_map(mag, title='real_x', save_dir=TEST_DIR)

audio_out = inverse_fn(mag, phase, spec_format=0)

output_filename = os.path.join(TEST_DIR, f'clear_{os.path.basename(inp)}')
write_audio(output_filename, audio_out, DEFAULT_SAMPLING_RATE)

generator_g = Generator(backbone_name='vgg16',
                        input_shape=(None, None, 3),
                        decoder_filters=(256, 128, 64, 32, 16),
                        classes=3,
                        activation='tanh')

discriminator_x = Discriminator(norm_type='instancenorm', target=False)

fake_x = generator_g(mag)
plot_heat_map(fake_x, title='fake_x', save_dir=TEST_DIR)

plot_heat_map(mel_fq(mag), title='mel_real_x', save_dir=TEST_DIR)

disc_real_x = discriminator_x(mel_fq(mag))
plot_heat_map(disc_real_x, title='disc_real_x', save_dir=TEST_DIR)

disc_fake_x = discriminator_x(mel_fq(fake_x))
plot_heat_map(disc_fake_x, title='disc_fake_x', save_dir=TEST_DIR)
