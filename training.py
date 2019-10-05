import numpy as np
from datetime import datetime

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from gtzan.struct import read_csv
from gtzan.model import build_model
from gtzan.generator import DataSequence

file_list = read_csv('/home/gtzan/jimmy/file_list.csv')
file_list2 = read_csv('/home/gtzan/jimmy/file_list2.csv')
exec_time = datetime.now().strftime('%Y%m%d%H%M%S')

train_generator = DataSequence(file_list, batch_size=8, shuffle=False)
test_generator = DataSequence(file_list2, batch_size=8, shuffle=False)

num_genres = 10

X = np.load(file_list[0][1])

input_shape = train_generator.input_shape

cnn = build_model(input_shape, num_genres)
cnn.compile(loss='sparse_categorical_crossentropy',
            optimizer=Adam(1e-5),
            metrics=['accuracy'])

checkpoint = ModelCheckpoint('vgg_model_v1.h5', monitor='val_loss', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)


hist = cnn.fit_generator(generator=train_generator,
                         validation_data=test_generator,
                         epochs=20,
                         callbacks=[checkpoint, earlystop],
                         use_multiprocessing=True,
                         workers=2)


