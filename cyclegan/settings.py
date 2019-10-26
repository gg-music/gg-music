MUSIC_NPY_PATH = {
    'guitar1': '/home/gtzan/data/gan_preprocessing/guitar1',
    'guitar2': '/home/gtzan/data/gan_preprocessing/guitar2',
    'piano1': '/home/gtzan/data/gan_preprocessing/piano1',
    'piano2': '/home/gtzan/data/gan_preprocessing/piano2',
    'cello': '/home/gtzan/data/gan_preprocessing/cello',
    'sax': '/home/gtzan/data/gan_preprocessing/sax'
}

MUSIC_SRC_PATH = {
    'wav': '/home/gtzan/data/gan/wav/sounds',
    'tfrecord': '/home/gtzan/data/gan_preprocessing/'
}

MUSIC_ROOT_PATH = '/home/gtzan/data'
MODEL_ROOT_PATH = '/home/gtzan/models'
EPOCHS = 100
DEFAULT_SAMPLING_RATE = 22050
DROPOUT_RATE = 0.5
BN_AXIS = 3
PAD_SIZE = ((0, 0), (8, 9))
STEPS = 400

X_INSTRUMENT = 'piano1'
Y_INSTRUMENT = 'guitar1'

INPUT_FILE = [['/home/gtzan/data/gan/wav/sounds/guitar1/guitar1-003.wav', 'f'],
              ['/home/gtzan/data/gan/wav/sounds/piano1/piano1-036.wav', 'g']]
