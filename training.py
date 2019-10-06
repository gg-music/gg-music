import numpy as np
from datetime import datetime

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from gtzan.model import build_model
from gtzan.generator import DataSequence
from gtzan.struct import load_fma
from gtzan.visdata import save_history

X, y = load_fma('/home/gtzan/data/fma_preprocessing')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

train_list = list(zip(X_train, y_train))
test_list = list(zip(X_test, y_test))

exec_time = datetime.now().strftime('%Y%m%d%H%M%S')

train_generator = DataSequence(train_list, batch_size=16, shuffle=False)
test_generator = DataSequence(test_list, batch_size=16, shuffle=False)

num_genres = 10

X = np.load(train_list[0][1])

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

save_history(hist, 'logs/{}/evaluate.png'.format(exec_time))
