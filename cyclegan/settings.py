from .helpers.utils import unet_padding_size

MUSIC_NPY_PATH = {
    'guitar1': '/home/gtzan/data/gan_preprocessing/guitar1',
    'guitar2': '/home/gtzan/data/gan_preprocessing/guitar2',
    'piano1': '/home/gtzan/data/gan_preprocessing/piano1',
    'piano2': '/home/gtzan/data/gan_preprocessing/piano2',

    'guitar1_cleaned': '/home/gtzan/data/gan_preprocessing/guitar1_cleaned',
    'guitar2_cleaned': '/home/gtzan/data/gan_preprocessing/guitar2_cleaned',
    'piano1_cleaned': '/home/gtzan/data/gan_preprocessing/piano1_cleaned',
    'piano2_cleaned': '/home/gtzan/data/gan_preprocessing/piano2_cleaned',
}
MUSIC_SRC_PATH = {
    'from': '/home/gtzan/data/gan/wav/sounds',
    'to': '/home/gtzan/data/gan_preprocessing/'
}
MODEL_ROOT_PATH = '/home/gtzan/models'
EPOCHS = 100
DEFAULT_SAMPLING_RATE = 22050
DROPOUT_RATE = 0.5
BN_AXIS = 3
PAD_SIZE = ((0,0), unet_padding_size(431, pool_size=2, layers=5))
