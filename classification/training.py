import sys
import os

workspace = os.path.dirname(os.getcwd())
sys.path.append(workspace)

from datetime import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from .helpers.learning_rate import CyclicLR
from .model.vgg_model import vgg16_model
from .helpers.data_generator import DataSequence
from .helpers.utils import get_file_list
from .helpers.plot import plot_save_history

X, y = get_file_list('/home/gtzan/ssd/fma_balanced', catalog_offset=-2)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

train_list = list(zip(X_train, y_train))
test_list = list(zip(X_test, y_test))

exec_time = datetime.now().strftime('%Y%m%d%H%M%S')

train_generator = DataSequence(train_list, batch_size=32, shuffle=False)
test_generator = DataSequence(test_list, batch_size=32, shuffle=False)

num_genres = 10

input_shape = train_generator.input_shape

mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    cnn = vgg16_model(input_shape, num_genres)
    cnn.compile(loss='sparse_categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

checkpoint = ModelCheckpoint('model/checkpoint_model_{}.h5'.format(exec_time),
                             monitor='val_loss',
                             save_best_only=True,
                             verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

clr = CyclicLR(base_lr=1e-5,
               max_lr=3e-4,
               step_size=len(train_generator) * 2,
               mode='triangular2')

hist = cnn.fit(train_generator,
               validation_data=test_generator,
               epochs=100,
               use_multiprocessing=True,
               shuffle=False,
               workers=10,
               verbose=1,
               callbacks=[clr, checkpoint, earlystop])

cnn.save('model/vgg_model_{}.h5'.format(exec_time))
print('save model', 'model/vgg_model_{}.h5'.format(exec_time))

plot_save_history(hist, 'logs/{}loss.png'.format(exec_time))
