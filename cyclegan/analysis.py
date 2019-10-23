import sys
import os
import numpy as np

workspace = os.path.dirname(os.getcwd())
sys.path.append(workspace)

from .helpers.plot import plot_stft

output_dir = "../logs"

cyc = [
    '/home/gtzan/data/gan_preprocessing/piano1/piano1-090.npy',
    '/home/gtzan/data/gan_preprocessing/piano1/piano1-033.npy',
    '/home/gtzan/data/gan_preprocessing/piano1/piano1-037.npy',
    '/home/gtzan/data/gan_preprocessing/piano1/piano1-041.npy',
    '/home/gtzan/data/gan_preprocessing/piano1/piano1-044.npy',
    '/home/gtzan/data/gan_preprocessing/piano1/piano1-047.npy',
    '/home/gtzan/data/gan_preprocessing/piano1/piano1-051.npy',
    '/home/gtzan/data/gan_preprocessing/piano1/piano1-063.npy',
    '/home/gtzan/data/gan_preprocessing/piano1/piano1-077.npy',
    '/home/gtzan/data/gan_preprocessing/piano1/piano1-081.npy'
]

for s in cyc:
    data = np.load(s)
    plot_stft(s)
