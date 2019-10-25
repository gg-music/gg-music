import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from . import signal
from .utils import make_dirs


def plot_epoch_loss(hist, save_dir, n_steps):
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(hist['gG'], label='generator_G_loss')
    plt.plot(hist['fG'], label='generator_F_loss')
    plt.title(f'Generator Loss-{n_steps}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist['xD'], label='discrminator_X_loss')
    plt.plot(hist['yD'], label='discrminator_y_loss')
    plt.title(f'Discrminator Loss-{n_steps}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    make_dirs(save_dir)
    output_path = os.path.join(save_dir, n_steps + '-loss.png')

    plt.tight_layout()
    plt.savefig(output_path, format='png', bbox_inches='tight')

# Plot and save keras trainning history
def plot_save_history(hist, save_dir):
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label='train')
    plt.plot(hist.history['val_accuracy'], label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir, format='png', bbox_inches='tight')


# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(save_dir,
                          cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_dir, format='png', bbox_inches='tight')


def plot_mfcc(npy, output_dir=None):
    song = np.load(npy)
    song_path, song_name = os.path.split(npy)
    song_path = song_path.split('/')
    title = os.path.join(song_path[-2], song_path[-1], song_name)
    save_dir = os.path.join(output_dir, song_name.split('.')[-2])

    s = song[0]
    s = s.reshape((128, 129))
    ax = sns.heatmap(s, robust=True, cbar=False)
    ax.set_title(title)
    ax.invert_yaxis()
    if output_dir:
        plt.savefig(save_dir + '.png', format='png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_stft(npy, output_dir=None):
    song = np.load(npy)
    song_path, song_name = os.path.split(npy)
    song_path = song_path.split('/')
    title = os.path.join(song_path[-2], song_path[-1], song_name)
    song = signal.amplitude_to_db(song)
    ax = sns.heatmap(song, vmin=0, vmax=0.5, cbar=False)
    ax.set_title(title)
    ax.invert_yaxis()
    if output_dir:
        save_dir = os.path.join(output_dir, song_name.split('.')[-2])
        plt.savefig(save_dir + '.png', format='png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_heat_map(img, title, save_dir=None):
    img = img[0, :, :, 0]
    img = (img + 1) / 2

    ax = sns.heatmap(img, vmin=0, vmax=1)
    ax.set_title(title)
    ax.invert_yaxis()

    make_dirs(save_dir)

    if save_dir:
        output = os.path.join(save_dir, title + '.png')
        plt.savefig(output, format='png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
