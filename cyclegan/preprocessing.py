import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .settings import MUSIC_SRC_PATH
from .helpers.utils import get_file_list, parallel_preprocessing
from .helpers.signal import to_stft

file_list = get_file_list(MUSIC_SRC_PATH['from'])
parallel_preprocessing(file_list,
                       MUSIC_SRC_PATH['to'],
                       spec_format=to_stft,
                       batch_size=10)
