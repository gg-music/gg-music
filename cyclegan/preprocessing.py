import os
import argparse
from functools import partial
from .helpers.parallel import batch_processing, processing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .settings import MUSIC_ROOT_PATH
from .helpers.utils import get_file_list
from .helpers.signal import to_stft
from .helpers.plot import plot_heat_map, plot_epoch_loss

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
        raise FileNotFoundError('Invalid src path')

    file_list = get_file_list(args.src_path)

    par = partial(batch_processing,
                  output_dir=os.path.join(MUSIC_ROOT_PATH,
                                          'gan_preprocessing/tfrecords'),
                  spec_format=to_stft,
                  to_tfrecord=args.tfrecord)

    processing(file_list, par, args.batch_size)
