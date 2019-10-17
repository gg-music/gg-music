from gtzan.utils import get_file_list, parallel_preprocessing, to_melspectrogram

src_path = '/home/gtzan/data/gtzan'
output_dir = '/home/gtzan/data/gtzan_preprocessing/'

file_list = get_file_list(src_path)

parallel_preprocessing(file_list, output_dir,
                       spec_format=to_melspectrogram,
                       batch_size=10, trim=30, split=0.1)
