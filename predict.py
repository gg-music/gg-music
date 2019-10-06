import json
import numpy as np
from scipy import stats
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import OrderedDict
from gtzan.visdata import plot_confusion_matrix
from gtzan.struct import get_file_list, load_mapping
from gtzan.generator import PredictSequence

from sklearn.metrics import confusion_matrix

exec_time = datetime.now().strftime('%Y%m%d%H%M%S')


X, y = get_file_list('/home/gtzan/ssd/gtzan_preprocessing', catalog_offset=-1)
y_val = np.array(y).astype(int)

predict_list = train_list = list(zip(X, y))
predict_generator = PredictSequence(predict_list, batch_size=16)

cnn = load_model('vgg_model_v1.h5')
pred = cnn.predict_generator(generator=predict_generator,
                             use_multiprocessing=True,
                             workers=2, verbose=1)

preds = np.argmax(pred, axis=1)

result = np.array(preds[:19000]).reshape((1000, 19))
mode = stats.mode(result, axis=1)
y_pred = np.array(mode[0]).reshape(1000,)


cm = confusion_matrix(y_pred, y_val)
print(cm)

keys = OrderedDict(sorted(load_mapping().items(), key=lambda t: t[1])).keys()
plot_confusion_matrix('logs/cm.png'.format(exec_time), cm, keys, normalize=True)
