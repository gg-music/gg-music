import os
import argparse
from functools import partial
from .helpers.parallel import batch_processing, processing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .settings import RAWDATA_ROOT_PATH
from .helpers.utils import get_file_list

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-s',
                    '--src_path',
                    required=True,
                    help='file src folder root path',
                    type=str)
    ap.add_argument('-b',
                    '--batch_size',
                    required=False,
                    default=10,
                    help='files per batch',
                    type=int)
    ap.add_argument('-cqt',
                    '--spectrum',
                    required=False,
                    default=False,
                    action='store_true',
                    help='convert to cqt, default is stft')
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

    destination = 'gan_preprocessing/tfrecords' if args.tfrecord else 'gan_preprocessing/npy'

    folder_suffix = 'cqt' if args.spectrum else 'stft'

    par = partial(batch_processing,
                  output_dir=os.path.join(RAWDATA_ROOT_PATH, destination),
                  spec_format=args.spectrum,
                  folder_suffix=folder_suffix,
                  to_tfrecord=args.tfrecord)

    processing(file_list, par, args.batch_size)
