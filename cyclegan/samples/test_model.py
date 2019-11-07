from cyclegan.segmentation_models.nestnet import Nestnet as Generator
from cyclegan.model.vgg_model import vgg16_model as Discriminator
from cyclegan.helpers.signal import preprocessing_fn, log_fq
from cyclegan.helpers.plot import plot_heat_map
import tensorflow as tf

inp = '/home/gtzan/data/gan_preprocessing/wav/piano3/piano3-0003.wav'
test_x, _ = preprocessing_fn(inp, spec_format=False)
test_x = test_x[tf.newaxis, :]

generator_g = Generator(backbone_name='vgg16',
                        input_shape=(None, None, 3),
                        decoder_filters=(256, 128, 64, 32, 16),
                        classes=3,
                        activation='tanh')

discriminator_x = Discriminator(norm_type='instancenorm', target=False)
print(test_x.shape)

plot_heat_map(test_x, title='real_x', save_dir='/home/gtzan/ssd/test')

plot_heat_map(log_fq(test_x), title='log_x', save_dir='/home/gtzan/ssd/test')

fake_x = generator_g(test_x)
plot_heat_map(fake_x, title='fake_x', save_dir='/home/gtzan/ssd/test')

disc_real_x = discriminator_x(log_fq(test_x))
plot_heat_map(disc_real_x, title='disc_real_x', save_dir='/home/gtzan/ssd/test')

disc_fake_x = discriminator_x(log_fq(fake_x))
plot_heat_map(disc_fake_x, title='disc_fake_x', save_dir='/home/gtzan/ssd/test')
