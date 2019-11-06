import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from .helpers.utils import get_file_list, make_dirs, check_rawdata_exists
from .helpers.example_protocol import extract_example
from .helpers.logger import save_loss_log, save_heatmap_npy
from .helpers.signal import preprocessing_fn
from .model_settings import *
from .settings import EPOCHS, MODEL_ROOT_PATH, STEPS, RAWSET_PATH

ap = argparse.ArgumentParser()
ap.add_argument('-m',
                '--model',
                required=True,
                help='your model name',
                type=str)
ap.add_argument('-x', required=True, help='convert from', type=str)
ap.add_argument('-y', required=True, help='convert to', type=str)
ap.add_argument('-tx', required=False, help='x_test wav', type=str)
ap.add_argument('-ty', required=False, help='y_test_wav', type=str)
ap.add_argument('-cqt',
                '--spectrum',
                required=False,
                default=False,
                action='store_true',
                help='convert to cqt, default is stft')
args = ap.parse_args()

x_rawset_path = os.path.join(RAWSET_PATH, args.x)
y_rawset_path = os.path.join(RAWSET_PATH, args.y)
check_rawdata_exists(x_rawset_path, y_rawset_path)

SAVE_MODEL_PATH = os.path.join(MODEL_ROOT_PATH, os.path.basename(args.model))
SAVE_DB_PATH = os.path.join(SAVE_MODEL_PATH, 'db')
SAVE_G_LOSS_PATH = os.path.join(SAVE_DB_PATH, 'Generator_loss')
SAVE_D_LOSS_PATH = os.path.join(SAVE_DB_PATH, 'Discriminator_loss')

make_dirs(SAVE_DB_PATH)
make_dirs(SAVE_G_LOSS_PATH)
make_dirs(SAVE_D_LOSS_PATH)

x_instrument, y_instrument = args.x, args.y

x_list = get_file_list(x_rawset_path)
y_list = get_file_list(y_rawset_path)

x_train_dataset = tf.data.TFRecordDataset(
    x_list[:STEPS]).prefetch(buffer_size=100).shuffle(buffer_size=100)
y_train_dataset = tf.data.TFRecordDataset(
    y_list[:STEPS]).prefetch(buffer_size=100).shuffle(buffer_size=100)

if args.tx and args.ty:
    tx = args.tx
    ty = args.ty
else:
    tx = args.x
    ty = args.y

x_test_path = os.path.join(os.path.dirname(RAWSET_PATH), 'wav', tx)
y_test_path = os.path.join(os.path.dirname(RAWSET_PATH), 'wav', ty)

x_test = get_file_list(x_test_path)[0]
y_test = get_file_list(y_test_path)[0]

test_x, _ = preprocessing_fn(x_test, args.spectrum)
test_y, _ = preprocessing_fn(y_test, args.spectrum)

test_x = test_x[np.newaxis, :]
test_y = test_y[np.newaxis, :]

save_heatmap_npy(test_x, '{}_reference'.format(x_instrument),
                 save_dir=os.path.join(SAVE_DB_PATH, 'fake_y'))
save_heatmap_npy(test_y, '{}_reference'.format(y_instrument),
                 save_dir=os.path.join(SAVE_DB_PATH, 'fake_x'))

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

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    last_epoch = len(ckpt_manager.checkpoints)
    print('Latest checkpoint epoch {} restored!!'.format(last_epoch))

start = len(ckpt_manager.checkpoints)
loss_history = {
    'Generator': {
        'f': [],
        'g': []
    },
    'Discriminator': {
        'x': [],
        'y': []
    }
}
for epoch in range(start, EPOCHS):

    start = time.time()
    n = 0
    pbar = tqdm(tf.data.Dataset.zip((x_train_dataset, y_train_dataset)),
                total=STEPS)
    for example_x, example_y in pbar:
        example_x = tf.train.Example.FromString(example_x.numpy())
        example_y = tf.train.Example.FromString(example_y.numpy())
        image_x = extract_example(example_x)
        image_y = extract_example(example_y)

        gG, fG, xD, yD = train_step(image_x['data'], image_y['data'], update='gd')

        loss_history['Generator']['g'].append(gG.numpy())
        loss_history['Generator']['f'].append(fG.numpy())
        loss_history['Discriminator']['x'].append(xD.numpy())
        loss_history['Discriminator']['y'].append(yD.numpy())

        if n % 10 == 0:
            # generate fake
            fake_y = generator_g(test_x)
            save_heatmap_npy(
                fake_y,
                '{}_epoch{:0>2}_step{:0>4}'.format(x_instrument, epoch + 1, n),
                os.path.join(SAVE_DB_PATH, 'fake_y'))
            fake_x = generator_f(test_y)
            save_heatmap_npy(
                fake_x,
                '{}_epoch{:0>2}_step{:0>4}'.format(y_instrument, epoch + 1, n),
                os.path.join(SAVE_DB_PATH, 'fake_x'))

            # disc fake
            disc_fake_y = discriminator_y(fake_y)
            save_heatmap_npy(
                disc_fake_y,
                'disc_fake_y_epoch{:0>2}_step{:0>4}'.format(epoch + 1, n),
                os.path.join(SAVE_DB_PATH, 'disc_fake_y'))

            disc_fake_x = discriminator_x(fake_x)
            save_heatmap_npy(
                disc_fake_x,
                'disc_fake_x_epoch{:0>2}_step{:0>4}'.format(epoch + 1, n),
                os.path.join(SAVE_DB_PATH, 'disc_fake_x'))

            # disc real
            disc_real_y = discriminator_y(test_y)
            save_heatmap_npy(
                disc_real_y,
                'disc_real_y_epoch{:0>2}_step{:0>4}'.format(epoch + 1, n),
                os.path.join(SAVE_DB_PATH, 'disc_real_y'))

            disc_real_x = discriminator_x(test_x)
            save_heatmap_npy(
                disc_real_x,
                'disc_real_x_epoch{:0>2}_step{:0>4}'.format(epoch + 1, n),
                os.path.join(SAVE_DB_PATH, 'disc_real_x'))

            # save loss
            save_loss_log(loss_history['Generator'], SAVE_G_LOSS_PATH, n,
                          epoch + 1, )
            save_loss_log(loss_history['Discriminator'], SAVE_D_LOSS_PATH, n,
                          epoch + 1)

        n += 1
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time() - start))
