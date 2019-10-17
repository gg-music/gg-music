from gtzan.utils import parallel_preprocessing, get_file_list
from gtzan.signal import to_melspectrogram
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='fma mp3 to npy')
parser.add_argument('-c', '--csv_folder', help='csv folder location', type=str, required=True)
parser.add_argument('-o', '--output_folder', help='output folder location', type=str, required=True)
args = parser.parse_args()

csv_dir = args.csv_folder
output_dir = args.output_folder

csv_list = get_file_list(csv_dir)

for csv in csv_list:
    file_list = pd.read_csv(csv, header=0)
    category = os.path.basename(csv).split('.')[-2]
    parallel_preprocessing(file_list['path'], output_dir,
                           category=category,
                           spec_format=to_melspectrogram,
                           batch_size=10, trim=30, split=0.1)
