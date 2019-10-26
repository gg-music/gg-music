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
