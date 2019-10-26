import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import tensorflow as tf
from tqdm import tqdm

from .helpers.utils import get_file_list, make_dirs
from .helpers.example_protocol import extract_example
from .helpers.plot import plot_heat_map, plot_epoch_loss
from .model_settings import *
from .settings import MUSIC_NPY_PATH, EPOCHS, MODEL_ROOT_PATH, STEPS, X_INSTRUMENT, Y_INSTRUMENT

ap = argparse.ArgumentParser()
ap.add_argument('-m',
                '--model',
                required=True,
                help='your model name',
                type=str)
args = ap.parse_args()

SAVE_MODEL_PATH = os.path.join(MODEL_ROOT_PATH, os.path.basename(args.model))

make_dirs(SAVE_MODEL_PATH)

x_instrument, y_instrument = X_INSTRUMENT, Y_INSTRUMENT

x_list = get_file_list(MUSIC_NPY_PATH[x_instrument])
y_list = get_file_list(MUSIC_NPY_PATH[y_instrument])

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

    plot_heat_map(test_x['data'],
                  title='{}_reference'.format(x_instrument),
                  save_dir=os.path.join(SAVE_MODEL_PATH, 'Generator_g'))

    plot_heat_map(test_y['data'],
                  title='{}_reference'.format(y_instrument),
                  save_dir=os.path.join(SAVE_MODEL_PATH, 'Generator_f'))

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
                                    update='gfd')

        loss_history['Generator']['g'].append(gG.numpy())
        loss_history['Generator']['f'].append(fG.numpy())
        loss_history['Discriminator']['x'].append(xD.numpy())
        loss_history['Discriminator']['y'].append(yD.numpy())

        if n % 10 == 0:
            prediction_g = generator_g(test_x['data'])
            plot_heat_map(
                prediction_g,
                '{}_epoch{:0>2}_step{:0>4}'.format(x_instrument,epoch + 1, n),
                os.path.join(SAVE_MODEL_PATH, 'Generator_g'))
            prediction_f = generator_f(test_y['data'])
            plot_heat_map(
                prediction_f,
                '{}_epoch{:0>2}_step{:0>4}'.format(y_instrument,epoch + 1, n),
                os.path.join(SAVE_MODEL_PATH, 'Generator_f'))
            prediction_y = discriminator_y(prediction_g)
            plot_heat_map(
                prediction_y,
                '{}_epoch{:0>2}_step{:0>4}'.format(y_instrument,epoch + 1, n),
                os.path.join(SAVE_MODEL_PATH, 'Discriminator_y'))
            prediction_x = discriminator_x(prediction_f)
            plot_heat_map(
                prediction_x,
                '{}_epoch{:0>2}_step{:0>4}'.format(x_instrument,epoch + 1, n),
                os.path.join(SAVE_MODEL_PATH, 'Discriminator_x'))

            plot_epoch_loss(loss_history, SAVE_MODEL_PATH, n, epoch + 1)

        n += 1
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time() - start))
