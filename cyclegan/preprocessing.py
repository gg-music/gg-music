import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from functools import partial
from .helpers.parallel import batch_processing, processing
from .helpers.utils import get_file_list
from .settings import RAWSET_PATH

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
                    default=100,
                    help='files per batch',
                    type=int)
    args = ap.parse_args()

    if not os.path.isdir(args.src_path):
        raise FileNotFoundError('Invalid src path')

    file_list = get_file_list(args.src_path)

    par = partial(batch_processing,
                  output_dir=os.path.join(RAWSET_PATH))

    processing(file_list, par, args.batch_size)
