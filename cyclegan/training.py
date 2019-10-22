from datetime import datetime
import time
import os
import argparse
import tensorflow as tf

from gtzan.data_generator import GanSequence
from gtzan.utils import get_file_list
from gtzan.losses import generator_loss, calc_cycle_loss, identity_loss, discriminator_loss
from gtzan.plot import plot_heat_map
from cyclegan.model_settings import *
from cyclegan.settings import MUSIC_NPY_PATH, CHECKPOINT_PATH, LOG_PATH, EPOCHS


ap = argparse.ArgumentParser()
ap.add_argument('-m','--load_model_name', required=False)
args = ap.parse_args()

if args.load_model_name:
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, args.load_model_name)
    LOG_PATH = os.path.join(LOG_PATH, args.load_model_name)
else:
    exec_time = datetime.now().strftime('%Y%m%d%H%M%S')
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, exec_time)
    os.mkdir(CHECKPOINT_PATH)
    LOG_PATH = os.path.join(LOG_PATH, exec_time)
    os.mkdir(LOG_PATH)


piano_train_list = get_file_list(MUSIC_NPY_PATH['piano1_cleaned'])
guitar_train_list = get_file_list(MUSIC_NPY_PATH['guitar1_cleaned'])
piano_test_list = get_file_list(MUSIC_NPY_PATH['piano2_cleaned'])
guitar_test_list = get_file_list(MUSIC_NPY_PATH['guitar2_cleaned'])

piano_data_gen = GanSequence(piano_train_list, batch_size=1, shuffle=True)
guitar_data_gen = GanSequence(guitar_train_list, batch_size=1, shuffle=True)
piano_test_gen = GanSequence(piano_test_list, batch_size=1, shuffle=True)
guitar_test_gen = GanSequence(guitar_test_list, batch_size=1, shuffle=True)

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=100)


# if a checkpoint exists, restore the latest checkpoint.

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    last_epoch = len(ckpt_manager.checkpoints)
    print('Latest checkpoint restored!! Last epoch {}'.format(last_epoch))


def generate_images(model, test_input, title, save_dir=None):
    prediction = model(test_input)
    plot_heat_map(prediction[0], title, save_dir)


@tf.function
def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
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
    with tf.device("/gpu:0"):
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              generator_g.trainable_variables)

    with tf.device("/gpu:1"):
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              generator_f.trainable_variables)

    with tf.device("/cpu:0"):
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


start = len(ckpt_manager.checkpoints)
for epoch in range(start, EPOCHS):
    plot_heat_map(piano_test_gen[0][0,:,:,:],
                  title='piano_reference',
                  save_dir=os.path.join(LOG_PATH, 'piano_to_guitar'))

    plot_heat_map(guitar_test_gen[0][0,:,:,:],
                  title='guitar_reference',
                  save_dir=os.path.join(LOG_PATH, 'guitar_to_piano'))
    start = time.time()


    n = 0
    for image_x, image_y in zip(piano_data_gen, guitar_data_gen):
        train_step(image_x, image_y)
        generate_images(generator_g, piano_test_gen[0],
                        title='epoch{:0>2}_step{:0>4}'.format(epoch+1, n),
                        save_dir=os.path.join(LOG_PATH,'piano_to_guitar'))

        generate_images(generator_f, guitar_test_gen[0],
                        title='epoch{:0>2}_step{:0>4}'.format(epoch+1, n),
                        save_dir=os.path.join(LOG_PATH,'guitar_to_piano'))

        if n % 10 == 0:
            print("epoch {} step {}".format(epoch+1, n))

        n += 1
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time() - start))
