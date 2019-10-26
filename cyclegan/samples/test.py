from cyclegan.segmentation_models.nestnet import Nestnet as Generator
from cyclegan.model.vgg_model import vgg16_model as Discriminator
# from cyclegan.model.pix2pix import discriminator as Discriminator
from cyclegan.helpers.plot import plot_heat_map
from cyclegan.helpers.example_protocol import extract_example
import tensorflow as tf

inp = '/home/gtzan/data/gan_preprocessing/guitar1/guitar1-018.tfrecords'
x_test_dataset = tf.data.TFRecordDataset(inp)

generator_g = Generator(backbone_name='vgg16',
                        input_shape=(None, None, 3),
                        decoder_filters=(256, 128, 64, 32, 16),
                        classes=3,
                        activation='tanh')

discriminator_x = Discriminator(norm_type='instancenorm', target=False)

for example_x in x_test_dataset:
    example_x = tf.train.Example.FromString(example_x.numpy())
    test_x = extract_example(example_x)
    fake_x = generator_g(test_x['data'])
    plot_heat_map(fake_x, title='fake_x')
    disc = discriminator_x(fake_x)
    plot_heat_map(disc, title='disc')
