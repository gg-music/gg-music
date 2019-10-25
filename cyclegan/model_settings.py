import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from .segmentation_models.nestnet import Nestnet as Generator
from .model.pix2pix import discriminator as Discriminator
from .helpers.losses import generator_loss, calc_cycle_loss, identity_loss, discriminator_loss

# Generator G translates X -> Y
# Generator F translates Y -> X.

with tf.device("/gpu:1"):
    generator_g = Generator(backbone_name='vgg16',
                            input_shape=(None, None, 3),
                            decoder_filters=(256, 128, 64, 32, 16),
                            classes=3,
                            activation='tanh')

    generator_f = Generator(backbone_name='vgg16',
                            input_shape=(None, None, 3),
                            decoder_filters=(256, 128, 64, 32, 16),
                            classes=3,
                            activation='tanh')

    discriminator_x = Discriminator(norm_type='instancenorm', target=False)
    discriminator_y = Discriminator(norm_type='instancenorm', target=False)

    generator_g_optimizer = Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = Adam(2e-4, beta_1=0.5)


@tf.function
def train_step(real_x, real_y, update='gfd'):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        with tf.device('/gpu:0'):
            fake_y = generator_g(real_x, training=True)
            cycled_x = generator_f(fake_y, training=True)

        with tf.device('/gpu:1'):
            fake_x = generator_f(real_y, training=True)
            cycled_y = generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = generator_f(real_x, training=True)
            same_y = generator_g(real_y, training=True)

        with tf.device("/gpu:0"):
            disc_real_x = discriminator_x(real_x, training=True)
            disc_real_y = discriminator_y(real_y, training=True)

            disc_fake_x = discriminator_x(fake_x, training=True)
            disc_fake_y = discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)

            total_cycle_loss = calc_cycle_loss(
                real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(
                real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(
                real_x, same_x)

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
    # Calculate the gradients for generator and discriminator
    if 'g' in update:
        with tf.device('/gpu:1'):
            generator_g_gradients = tape.gradient(
                total_gen_g_loss, generator_g.trainable_variables)
    if 'f' in update:
        with tf.device("/gpu:0"):
            generator_f_gradients = tape.gradient(
                total_gen_f_loss, generator_f.trainable_variables)

    if 'd' in update:
        with tf.device('/gpu:1'):
            discriminator_x_gradients = tape.gradient(
                disc_x_loss, discriminator_x.trainable_variables)
            discriminator_y_gradients = tape.gradient(
                disc_y_loss, discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
    if 'g' in update:
        generator_g_optimizer.apply_gradients(
            zip(generator_g_gradients, generator_g.trainable_variables))
    if 'f' in update:
        generator_f_optimizer.apply_gradients(
            zip(generator_f_gradients, generator_f.trainable_variables))

    if 'd' in update:
        discriminator_x_optimizer.apply_gradients(
            zip(discriminator_x_gradients, discriminator_x.trainable_variables))

        discriminator_y_optimizer.apply_gradients(
            zip(discriminator_y_gradients, discriminator_y.trainable_variables))

        return gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss
