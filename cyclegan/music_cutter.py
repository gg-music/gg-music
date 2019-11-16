import os
import time
import argparse
from pydub import AudioSegment

parser = argparse.ArgumentParser(
    description='cut raw music into "n" seconds/part')
parser.add_argument('-s',
                    '--seconds',
                    help='n seconds per part',
                    type=float,
                    required=True)
parser.add_argument('-i',
                    '--input',
                    help='input file location',
                    type=str,
                    required=True)
parser.add_argument('--suffix',
                    help='add preprocessing suffix',
                    type=str,
                    required=False)
args = parser.parse_args()
try:
    raw_audio = AudioSegment.from_wav(args.input)
except Exception as e:
    raise e

spilit_time = args.seconds * 1000
folder_name = args.input.split(
    '/')[-1][0:-4] if not args.suffix else args.input.split(
    '/')[-1][0:-4] + '_' + args.suffix

output_dir = f'/home/gtzan/data/gan_preprocessing/wav/{folder_name}'

if not os.path.isdir(output_dir):
    os.makedirs(output_dir, mode=0o777)

for i in range(0, 100000000000):
    begin = spilit_time * i
    end = begin + spilit_time
    new_audio = raw_audio[begin:end]

    if len(new_audio) < spilit_time:
        break

    file_name = '{}{}{:04d}{}'.format(folder_name, '-', i, '.wav')
    print(file_name)
    new_audio.export(output_dir + '/' + file_name, format="wav")

    time.sleep(0.5)
