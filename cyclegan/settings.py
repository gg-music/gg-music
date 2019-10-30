RAWSET_PATH = '/home/gtzan/data/gan_preprocessing/tfrecords'
MUSIC_SRC_PATH = {
    'wav': '/home/gtzan/data/gan/wav/sounds',
    'tfrecord': '/home/gtzan/data/gan_preprocessing/'
}

RAWDATA_ROOT_PATH = '/home/gtzan/data'
MODEL_ROOT_PATH = '/home/gtzan/models'
EPOCHS = 100
DEFAULT_SAMPLING_RATE = 22050
DROPOUT_RATE = 0.5
BN_AXIS = 3
PAD_SIZE = ((0, 0), (1, 0))
STEPS = 400

INPUT_FILE = [['/home/gtzan/data/gan/wav/sax/sax-001.wav', 'f'],
              ['/home/gtzan/data/gan/wav/cello/cello-005.wav', 'g']]
