import numpy as np
from scipy import stats
import argparse

import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import OrderedDict
from gtzan.visdata import plot_confusion_matrix
from gtzan.struct import get_file_list, load_mapping
from gtzan.generator import PredictSequence

from sklearn.metrics import confusion_matrix, accuracy_score

parser = argparse.ArgumentParser(description='predict gtzan')
parser.add_argument('-m', '--model', help='model', type=str, required=True)
args = parser.parse_args()

X, y = get_file_list('/home/gtzan/ssd/gtzan_preprocessing', catalog_offset=-1)
y_val = np.array(y).astype(int)

predict_list = train_list = list(zip(X, y))
predict_generator = PredictSequence(predict_list, batch_size=16)

cnn = load_model(args.model)

pred = cnn.predict(predict_generator,
                   use_multiprocessing=True,
                   workers=5, verbose=1)

preds = np.argmax(pred, axis=1)
result = np.array(preds[:19000]).reshape((1000, 19))
mode = stats.mode(result, axis=1)
y_pred = np.array(mode[0]).reshape(1000, )

cm = confusion_matrix(y_val, y_pred)
print(cm)

acc = accuracy_score(y_val, y_pred)
print('acc= ', acc)

keys = OrderedDict(sorted(load_mapping(reverse=True).items(), key=lambda t: t[1])).keys()
plot_confusion_matrix('logs/{}cm.png'.format(args.model), cm, keys, normalize=True)
