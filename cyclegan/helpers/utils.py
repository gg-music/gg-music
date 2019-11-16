import os


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


def load_model(model_path, n_epoch):
    import tensorflow as tf
    from cyclegan.model.model_settings import generator_g, generator_f

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                               generator_f=generator_f)

    ckpt_manager = tf.train.CheckpointManager(ckpt, model_path, max_to_keep=100)
    last_epoch = len(ckpt_manager.checkpoints)

    if n_epoch:
        epoch = n_epoch
        ckpt.restore(ckpt_manager.checkpoints[epoch - 1]).expect_partial()
        print('Checkpoint epoch {} restored!!'.format(epoch))
    else:
        epoch = last_epoch
        ckpt.restore(ckpt_manager.checkpoints[epoch - 1]).expect_partial()
        print('Latest checkpoint epoch {} restored!!'.format(epoch))

    models = {'g': generator_g, 'f': generator_f}

    return ckpt, ckpt_manager, models, epoch
