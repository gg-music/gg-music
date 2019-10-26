import os
import argparse
from functools import partial
from .helpers.parallel import batch_preprocessing, preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .settings import MUSIC_SRC_PATH, MUSIC_ROOT_PATH
from .helpers.utils import get_file_list
from .helpers.signal import to_stft

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-s',
                    '--src_path',
                    required=True,
                    help='file src folder root path',
                    type=str)
    ap.add_argument('--batch_size',
                    required=False,
                    default=10,
                    help='files per batch',
                    type=int)
    ap.add_argument('-tf',
                    '--tfrecord',
                    required=False,
                    default=False,
                    action='store_true',
                    help='convert to tfrecord')

    args = ap.parse_args()

    if not os.path.isdir(args.src_path):
        raise NotADirectoryError('Invalid src path')

    file_list = get_file_list(args.src_path)

    par = partial(batch_preprocessing,
                  output_dir=MUSIC_ROOT_PATH,
                  spec_format=to_stft,
                  to_tfrecord=args.tfrecord)

    preprocessing(file_list, par, args.batch_size)
