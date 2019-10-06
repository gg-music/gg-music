from gtzan.struct import get_file_list, parallel_preprocessing

src_path = '/home/gtzan/data/gtzan'
output_dir = '/home/gtzan/data/gtzan_preprocessing/'

file_list = get_file_list(src_path)

parallel_preprocessing(file_list, output_dir, batch_size=10)
