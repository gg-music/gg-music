RAWSET_PATH = '/home/gtzan/data/gan_preprocessing/tfrecords'
MUSIC_SRC_PATH = {
    'wav': '/home/gtzan/data/gan/wav/sounds',
    'tfrecord': '/home/gtzan/data/gan_preprocessing/'
}

RAWDATA_ROOT_PATH = '/home/gtzan/data'
MODEL_ROOT_PATH = '/home/gtzan/ssd/models'
EPOCHS = 20
DEFAULT_SAMPLING_RATE = 22050
DROPOUT_RATE = 0.5
BN_AXIS = 3
PAD_SIZE = ((0, 0), (1, 0))
STEPS = 2500

INPUT_FILE = [['/home/gtzan/data/gan_preprocessing/wav/guitar3/guitar1-0001.wav', 'f'],
              ['/home/gtzan/data/gan_preprocessing/wav/piano1/piano1-0001.wav', 'g'], ]
