from gtzan.utils import get_file_list, parallel_preprocessing, to_stft

src_path = '/home/gtzan/data/gan/wav/sounds'
output_dir = '/home/gtzan/data/gan_preprocessing/'

file_list = get_file_list(src_path)
parallel_preprocessing(file_list, output_dir, spec_format=to_stft, batch_size=10)
