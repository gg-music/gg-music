import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from gtzan.model import build_model
from gtzan.generator import DataSequence
from gtzan.struct import get_file_list
from gtzan.visdata import save_history

X, y = get_file_list('/home/gtzan/ssd/fma_preprocessing', catalog_offset=-2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42, stratify=y)

train_list = list(zip(X_train, y_train))
test_list = list(zip(X_test, y_test))

exec_time = datetime.now().strftime('%Y%m%d%H%M%S')

train_generator = DataSequence(train_list, batch_size=16, shuffle=False)
test_generator = DataSequence(test_list, batch_size=16, shuffle=False)

num_genres = 10

input_shape = train_generator.input_shape

mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    cnn = build_model(input_shape, num_genres)
    cnn.compile(loss='sparse_categorical_crossentropy',
                optimizer=Adam(1e-4),
                metrics=['accuracy'])

hist = cnn.fit(train_generator,
               epochs=1,
               use_multiprocessing=True,
               shuffle=False,
               workers=10, verbose=1)


cnn.save('vgg_model_{}.h5'.format(exec_time))
print('save model','vgg_model_{}.h5'.format(exec_time))
