from .utils import make_dirs
import os
from pandas.io.json._normalize import nested_to_record
import numpy as np


def save_loss_log(hist, save_dir, n_steps, n_epoch, delimiter=','):
    """
        n_steps/per file
    """
    flat_hist = nested_to_record(hist, sep='_')
    for model, npy in flat_hist.items():
        folder = os.path.join(save_dir, model)

        make_dirs(folder)

        filename = f'epoch{n_epoch:02}_{n_steps:04}-loss.log'

        with open(os.path.join(folder, filename), 'w') as fp:
            for n in npy:
                fp.write("{}{}".format(n, delimiter))


def save_heatmap_npy(img, title, save_dir=None):
    make_dirs(save_dir)
    np.save(file=os.path.join(save_dir, title), arr=img)
