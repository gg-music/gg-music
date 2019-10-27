import time
from pydub import AudioSegment
import argparse
import os

parser = argparse.ArgumentParser(
    description='cut raw music into "n" seconds/part')
parser.add_argument('-s',
                    '--seconds',
                    help='n seconds per part',
                    type=float,
                    required=True)
parser.add_argument('-f',
                    '--file',
                    help='input file location',
                    type=str,
                    required=True)

args = parser.parse_args()
try:
    raw_audio = AudioSegment.from_wav(args.file)
except Exception as e:
    raise e

spilit_time = args.seconds * 1000
prefix = args.file.split('/')[-1][0:-4]

output_dir = f'/home/gtzan/data/gan_preprocessing/wav/{prefix}'

if not os.path.isdir(output_dir):
    os.makedirs(output_dir, mode=0o777)

for i in range(0, 100000000000):
    begin = spilit_time * i
    end = begin + spilit_time
    new_audio = raw_audio[begin:end]

    if (len(new_audio) < spilit_time):
        break

    file_name = '{}{}{:04d}{}'.format(prefix, '-', i, '.wav')
    print(file_name)
    new_audio.export(output_dir + '/' + file_name, format="wav")

    time.sleep(0.5)
