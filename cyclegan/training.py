from datetime import datetime
import time
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from gtzan.data_generator import GanSequence
from gtzan.utils import get_file_list, unet_padding_size, crop
from gtzan.segmentation_models.unet import Unet as Generator
from gtzan.model.pix2pix import discriminator as Discriminator
from gtzan.losses import generator_loss, calc_cycle_loss, identity_loss, discriminator_loss
from gtzan.plot import plot_heat_map

exec_time = datetime.now().strftime('%Y%m%d%H%M%S')

guitar = get_file_list('/home/gtzan/data/gan_preprocessing/guitar1')
piano = get_file_list('/home/gtzan/data/gan_preprocessing/piano1')

guitar_data_gen = GanSequence(guitar, batch_size=1, shuffle=False)
piano_data_gen = GanSequence(piano, batch_size=1, shuffle=False)

PAD_SIZE = ((0, 0), unet_padding_size(guitar_data_gen.input_shape[1], pool_size=2, layers=8))
IPT_SHAPE = [guitar_data_gen.input_shape[0], guitar_data_gen.input_shape[1] + PAD_SIZE[1][0] + PAD_SIZE[1][1]]

generator_g = Generator(backbone_name='vgg16',
                        input_shape=(None, None, 3),
                        decoder_filters=(512, 512, 256, 128, 64),
                        classes=3,
                        activation='tanh')

generator_f = Generator(backbone_name='vgg16',
                        input_shape=(None, None, 3),
                        decoder_filters=(512, 512, 256, 128, 64),
                        classes=3,
                        activation='tanh')

# generator_g = Generator(3, norm_type='instancenorm')
# generator_f = Generator(3, norm_type='instancenorm')

discriminator_x = Discriminator(norm_type='instancenorm', target=False)
discriminator_y = Discriminator(norm_type='instancenorm', target=False)

generator_g_optimizer = Adam(2e-4, beta_1=0.5)
generator_f_optimizer = Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = Adam(2e-4, beta_1=0.5)

checkpoint_path = "/home/gtzan/jimmy/model"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


def generate_images(model, test_input):
    prediction = model(test_input)
    plot_heat_map(prediction[0])


@tf.function
def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        print('tape')
        # Generator G translates X -> Y
        # Generator F translates Y -> X.
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(
            real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(
            real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(
            real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(
        disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(
        disc_y_loss, discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(
        zip(generator_g_gradients, generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(
        zip(generator_f_gradients, generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(
        zip(discriminator_x_gradients, discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(
        zip(discriminator_y_gradients, discriminator_y.trainable_variables))


EPOCHS = 40
for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_x, image_y in zip(piano_data_gen, guitar_data_gen):
        train_step(image_x, image_y)
        if n % 10 == 0:
            print('.', end='')
        n += 1

        generate_images(generator_g, piano_data_gen[0])

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(
            epoch + 1, ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time() - start))
