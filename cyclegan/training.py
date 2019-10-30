import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import tensorflow as tf
from tqdm import tqdm

from .helpers.utils import get_file_list, make_dirs, check_rawdata_exists
from .helpers.example_protocol import extract_example
from .helpers.logger import save_loss_log, save_heatmap_npy
from .helpers.plot import plot_heat_map, plot_epoch_loss
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
args = ap.parse_args()

x_rawset_path = os.path.join(RAWSET_PATH, args.x)
y_rawset_path = os.path.join(RAWSET_PATH, args.y)
check_rawdata_exists(x_rawset_path, y_rawset_path)

SAVE_MODEL_PATH = os.path.join(MODEL_ROOT_PATH, os.path.basename(args.model))
SAVE_NPY_PATH = os.path.join(SAVE_MODEL_PATH, 'npy')
SAVE_LOG_PATH = os.path.join(SAVE_MODEL_PATH, 'logs')

make_dirs(SAVE_MODEL_PATH)
make_dirs(SAVE_LOG_PATH)

x_instrument, y_instrument = args.x, args.y

x_list = get_file_list(x_rawset_path)
y_list = get_file_list(y_rawset_path)

x_train_dataset = tf.data.TFRecordDataset(
    x_list[:STEPS]).prefetch(buffer_size=100)
y_train_dataset = tf.data.TFRecordDataset(
    y_list[:STEPS]).prefetch(buffer_size=100)

x_test_dataset = tf.data.TFRecordDataset(x_list[STEPS])
y_test_dataset = tf.data.TFRecordDataset(y_list[STEPS])

for example_x, example_y in tf.data.Dataset.zip(
    (x_test_dataset, y_test_dataset)):
    example_x = tf.train.Example.FromString(example_x.numpy())
    example_y = tf.train.Example.FromString(example_y.numpy())
    test_x = extract_example(example_x)
    test_y = extract_example(example_y)
    save_heatmap_npy(
    # plot_heat_map(
                     test_x['data'],
                     '{}_reference'.format(x_instrument),
                     save_dir=os.path.join(SAVE_NPY_PATH, 'test/Generator_g'))
    save_heatmap_npy(
    # plot_heat_map(
                     test_y['data'],
                     '{}_reference'.format(y_instrument),
                     save_dir=os.path.join(SAVE_NPY_PATH, 'test/Generator_f'))

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

        gG, fG, xD, yD = train_step(image_x['data'],
                                    image_y['data'],
                                    update='gd')
        train_step(image_x['data'], image_y['data'], update='d')

        loss_history['Generator']['g'].append(gG.numpy())
        loss_history['Generator']['f'].append(fG.numpy())
        loss_history['Discriminator']['x'].append(xD.numpy())
        loss_history['Discriminator']['y'].append(yD.numpy())

        if n % 10 == 0 and n != 0:
            prediction_g = generator_g(test_x['data'])
            save_heatmap_npy(
            # plot_heat_map(
                prediction_g,
                '{}_epoch{:0>2}_step{:0>4}'.format(x_instrument, epoch + 1, n),
                os.path.join(SAVE_NPY_PATH, 'Generator_g'))
            prediction_f = generator_f(test_y['data'])
            save_heatmap_npy(
            # plot_heat_map(
                prediction_f,
                '{}_epoch{:0>2}_step{:0>4}'.format(y_instrument, epoch + 1, n),
                os.path.join(SAVE_NPY_PATH, 'Generator_f'))

            prediction_y = discriminator_y(prediction_g)
            save_heatmap_npy(
            # plot_heat_map(
                prediction_y,
                '{}_epoch{:0>2}_step{:0>4}'.format(y_instrument, epoch + 1, n),
                os.path.join(SAVE_NPY_PATH, 'Discriminator_y'))

            prediction_x = discriminator_x(prediction_f)
            save_heatmap_npy(
            # plot_heat_map(
                prediction_x,
                '{}_epoch{:0>2}_step{:0>4}'.format(x_instrument, epoch + 1, n),
                os.path.join(SAVE_NPY_PATH, 'Discriminator_x'))

            save_loss_log(loss_history, SAVE_LOG_PATH, n, epoch + 1)
            # plot_epoch_loss(loss_history, SAVE_LOG_PATH, n, epoch + 1)

        n += 1
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))

    print('Time taken for epoch {} is {} sec, total loss {}\n'.format(epoch + 1,
                                                       time.time() - start), )
