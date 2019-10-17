import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from gtzan.model import build_model
from gtzan.generator import DataSequence
from gtzan.utils import get_file_list
from gtzan.lr_finder import LRFinder
from gtzan.visdata import save_history

exec_time = datetime.now().strftime('%Y%m%d%H%M%S')

X, y = get_file_list('/home/gtzan/ssd/fma_balanced', catalog_offset=-2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
train_list = list(zip(X_train, y_train))

train_generator = DataSequence(train_list, batch_size=32, shuffle=False)

num_genres = 10
input_shape = train_generator.input_shape

mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    cnn = build_model(input_shape, num_genres)
    cnn.compile(loss='sparse_categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

lr_finder = LRFinder(min_lr=1e-6,
                     max_lr=1e-2,
                     steps_per_epoch=len(train_generator),
                     epochs=3)

cnn.fit(train_generator,
        shuffle=False,
        callbacks=[lr_finder])

lr_finder.plot_loss()