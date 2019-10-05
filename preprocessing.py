from gtzan.struct import read_csv, parallel_preprocessing
import os

src_csv = '/home/gtzan/gtzan_genre/reggae/602-reggae___dancehall.csv'
output_dir = '/home/gtzan/data/fma_preprocessing/reggae/'

catagory_name = os.path.basename(src_csv).split('.')[0]
catagory_dir = os.path.join(output_dir, catagory_name)
os.makedirs(catagory_dir, mode=0o777, exist_ok=True)


parallel_preprocessing(src_csv, catagory_dir, batch_size=10)

