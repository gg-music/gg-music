import sys
import os

workspace = os.path.dirname(os.getcwd())
sys.path.append(workspace)

import numpy as np
from scipy import stats
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import OrderedDict
from .helpers.plot import plot_confusion_matrix
from .helpers.utils import get_file_list, load_mapping, pred_to_y
from .helpers.data_generator import PredictSequence
from sklearn.metrics import confusion_matrix, accuracy_score

parser = argparse.ArgumentParser(description='predict gtzan')
parser.add_argument('-m', '--model', help='model', type=str, required=True)
args = parser.parse_args()

model_name = os.path.basename(args.model).split('.')[-2]

X, y = get_file_list('/home/gtzan/ssd/gtzan_preprocessing', catalog_offset=-1)
y_val = np.array(y).astype(int)

predict_list = train_list = list(zip(X, y))
predict_generator = PredictSequence(predict_list, batch_size=16)

cnn = load_model(args.model)

pred = cnn.predict(predict_generator,
                   use_multiprocessing=True,
                   workers=5,
                   verbose=1)

y_pred = pred_to_y(pred, n_song=len(y), split_per_song=predict_generator.dim[0])

cm = confusion_matrix(y_val, y_pred)
print(cm)

acc = accuracy_score(y_val, y_pred)
print('acc= ', acc)

keys = OrderedDict(
    sorted(load_mapping(reverse=True).items(), key=lambda t: t[1])).keys()
plot_confusion_matrix('logs/{}cm.png'.format(model_name),
                      cm,
                      keys,
                      normalize=True)
