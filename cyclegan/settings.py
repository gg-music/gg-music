RAWSET_PATH = '/home/gtzan/data/gan_preprocessing/tfrecords'
RAWDATA_ROOT_PATH = '/home/gtzan/data'
MODEL_ROOT_PATH = '/home/gtzan/ssd/models'
EPOCHS = 20
DEFAULT_SAMPLING_RATE = 22050
DROPOUT_RATE = 0.5
BN_AXIS = 3
PAD_SIZE = ((0, 0), (1, 0))
STEPS = 3000

INPUT_FILE = [['/home/gtzan/data/gan_preprocessing/wav/guitar4/guitar4-0377.wav', 'f'],
              ['/home/gtzan/data/gan_preprocessing/wav/piano1/piano1-0118.wav', 'g'], ]
