from datetime import datetime
import time
import os
import argparse
import tensorflow as tf

from .helpers.data_generator import GanSequence
from .helpers.utils import get_file_list, make_dirs
from .helpers.losses import generator_loss, calc_cycle_loss, identity_loss, discriminator_loss
from .helpers.plot import plot_heat_map
from .model_settings import *
from .settings import MUSIC_NPY_PATH, EPOCHS, MODEL_ROOT_PATH

ap = argparse.ArgumentParser()
ap.add_argument('-m',
                '--model',
                required=True,
                help='your model name',
                type=str)
args = ap.parse_args()

SAVE_MODEL_PATH = os.path.join(MODEL_ROOT_PATH, os.path.basename(args.model))
make_dirs(SAVE_MODEL_PATH)

piano_train_list = get_file_list(MUSIC_NPY_PATH['piano1_cleaned'])
guitar_train_list = get_file_list(MUSIC_NPY_PATH['guitar1_cleaned'])
piano_test_list = get_file_list(MUSIC_NPY_PATH['piano2_cleaned'])
guitar_test_list = get_file_list(MUSIC_NPY_PATH['guitar2_cleaned'])

piano_data_gen = GanSequence(piano_train_list, batch_size=1, shuffle=True)
guitar_data_gen = GanSequence(guitar_train_list, batch_size=1, shuffle=True)
piano_test_gen = GanSequence(piano_test_list, batch_size=1, shuffle=False)
guitar_test_gen = GanSequence(guitar_test_list, batch_size=1, shuffle=False)

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt,
                                          SAVE_MODEL_PATH,
                                          max_to_keep=100)

# if a checkpoint exists, restore the latest checkpoint.

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    last_epoch = len(ckpt_manager.checkpoints)
    print('Latest checkpoint epoch {} restored!!'.format(last_epoch))

start = len(ckpt_manager.checkpoints)
for epoch in range(start, EPOCHS):
    plot_heat_map(piano_test_gen[0][0, :, :, :],
                  title='piano_reference',
                  save_dir=os.path.join(SAVE_MODEL_PATH, 'piano_to_guitar'))

    plot_heat_map(guitar_test_gen[0][0, :, :, :],
                  title='guitar_reference',
                  save_dir=os.path.join(SAVE_MODEL_PATH, 'guitar_to_piano'))
    start = time.time()

    n = 0
    for image_x, image_y in zip(piano_data_gen, guitar_data_gen):
        train_step(image_x, image_y)

        prediction_g = generator_g(piano_test_gen[0])
        plot_heat_map(prediction_g[0],
                      'epoch{:0>2}_step{:0>4}'.format(epoch + 1, n),
                      os.path.join(SAVE_MODEL_PATH, 'piano_to_guitar'))
        prediction_f = generator_f(guitar_test_gen[0])
        plot_heat_map(prediction_f[0],
                      'epoch{:0>2}_step{:0>4}'.format(epoch + 1, n),
                      os.path.join(SAVE_MODEL_PATH, 'guitar_to_piano'))

        if n % 10 == 0:
            print("epoch {} step {}".format(epoch + 1, n))

        n += 1
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time() - start))
