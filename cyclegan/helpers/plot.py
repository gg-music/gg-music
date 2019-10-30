import os
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import make_dirs


def plot_epoch_loss_by_log(log, save_dir, title):
    plt.figure(figsize=(4, 4), dpi=100)
    plt.plot(log, label=title)
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.xlim(left=max(0, len(log) - 200), right=max(200, len(log)))
    plt.legend()

    make_dirs(save_dir)

    output_path = os.path.join(save_dir, f'{title}.png')
    plt.savefig(output_path, format='png', dpi=100)
    plt.close()


def plot_epoch_loss(hist, save_dir, n_steps, n_epoch):
    for model_type, models in hist.items():
        plt.figure(figsize=(4, 4), dpi=100)
        for model, npy in models.items():
            plt.plot(npy, label=f'{model_type}_{model}')

        plt.title(f'{model_type} Loss-{n_steps:04} Epoch-{n_epoch:02}')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.xlim(left=max(0, len(npy) - 200), right=max(200, len(npy)))
        plt.legend()

        make_dirs(f'{save_dir}/{model_type}')

        output_path = os.path.join(f'{save_dir}/{model_type}',
                                   f'epoch{n_epoch:02}_{n_steps:04}-loss.png')
        plt.savefig(output_path, format='png', dpi=100)
        plt.close()


def plot_heat_map(img, title, save_dir):
    img = img[0, :, :, 0]
    img = (img + 1) / 2
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax = sns.heatmap(img, vmin=0, vmax=1, ax=ax, cbar=False)
    ax.set_title(title)
    ax.invert_yaxis()

    if save_dir:
        make_dirs(save_dir)
        output = os.path.join(save_dir, title + '.png')
        plt.savefig(output, format='png', dpi=100)
        plt.close()
    else:
        plt.show()
