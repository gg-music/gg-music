import os
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import make_dirs


def plot_epoch_loss(hist, save_dir, n_steps):
    plt.figure(figsize=(8, 4), dpi=100)

    plt.subplot(1, 2, 1)
    plt.plot(hist['gG'], label='generator_G_loss')
    plt.plot(hist['fG'], label='generator_F_loss')
    plt.title(f'Generator Loss-{n_steps:04}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist['xD'], label='discrminator_X_loss')
    plt.plot(hist['yD'], label='discrminator_y_loss')
    plt.title(f'Discrminator Loss-{n_steps:04}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    make_dirs(save_dir)

    output_path = os.path.join(save_dir, f'{n_steps:04}-loss.png')

    plt.savefig(output_path, format='png', dpi=100)


def plot_heat_map(img, title, save_dir=None):
    img = img[0, :, :, 0]
    img = (img + 1) / 2

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax = sns.heatmap(img,
                     vmin=0,
                     vmax=1,
                     ax=ax,
                     cbar_kws={'orientation': 'horizontal'})
    ax.set_title(title)
    ax.invert_yaxis()

    make_dirs(save_dir)

    if save_dir:
        output = os.path.join(save_dir, title + '.png')
        plt.savefig(output, format='png', dpi=100)
        plt.close()
    else:
        plt.show()
