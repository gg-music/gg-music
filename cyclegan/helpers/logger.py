from .utils import make_dirs
import os
import numpy as np


def save_loss_log(hist, save_dir, n_steps, n_epoch):
    """
        n_steps/per file
    """
    model_type = os.path.basename(save_dir).split('_')[-2]
    for npy in hist.values():
        filename = f'{model_type}_epoch{n_epoch:02}_{n_steps:04}-loss.log'

        with open(os.path.join(save_dir, filename), 'a') as fp:
            for n in npy:
                fp.write(f'{n},')


def save_heatmap_npy(img, title, save_dir=None):
    make_dirs(save_dir)
    np.save(file=os.path.join(save_dir, title), arr=img)
