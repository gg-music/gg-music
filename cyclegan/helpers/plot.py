import os
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import make_dirs


def plot_epoch_loss_by_log(logs, save_dir, title):
    models = {'Generator': 'fg',
              'Discriminator': 'xy'}
    model_type = os.path.basename(save_dir).split('_')[-2]

    plt.figure(figsize=(4, 4), dpi=100)
    log_1st = logs[:logs.size // 2]
    log_2nd = logs[logs.size // 2:]
    plt.plot(log_1st, label=f'{model_type}_{models[model_type][0]}')
    plt.plot(log_2nd, label=f'{model_type}_{models[model_type][1]}')
    plt.title(f'{title}')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    left = max(0, len(log_1st) - 200)
    right = max(200, len(log_1st))
    top = max(max(log_1st[left:right]), max(log_2nd[left:right]))
    down = min(min(log_1st[left:right]), min(log_2nd[left:right]))
    bound = (top - down) * 0.1
    plt.xlim(left=left, right=right)
    plt.ylim(down-bound, top+bound)
    plt.xticks([])
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
    if 'disc' in title:
        fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    else:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax = sns.heatmap(img, vmin=0, vmax=1, ax=ax, cbar=False)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.axis('off')

    if save_dir:
        make_dirs(save_dir)
        output = os.path.join(save_dir, title + '.png')
        plt.savefig(output, format='png', dpi=100)
        plt.close()
    else:
        plt.show()
