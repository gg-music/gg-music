import os
import copy


def get_file_list(src_dir):
    input_path = []
    for dir_path, subdir, filenames in os.walk(src_dir):
        for f in filenames:
            input_path.append(os.path.join(dir_path, f))

    return input_path


def make_dirs(path):
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o777)


def check_rawdata_exists(path_x, path_y):
    if any((not os.path.isdir(path_x), not os.path.isdir(path_y))):
        raise FileNotFoundError("input instrument pair does not exists")


def load_model(n_epoch, ckpt, ckpt_manager):
    import tensorflow as tf
    last_epoch = len(ckpt_manager.checkpoints)

    if n_epoch:
        epoch = n_epoch
        ckpt.restore(ckpt_manager.checkpoints[epoch - 1]).expect_partial()
        print('Checkpoint epoch {} restored!!'.format(epoch))
    else:
        epoch = last_epoch
        ckpt.restore(ckpt_manager.checkpoints[epoch - 1]).expect_partial()
        print('Latest checkpoint epoch {} restored!!'.format(epoch))

    return ckpt_manager, epoch
