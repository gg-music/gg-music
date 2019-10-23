import sys
import os

workspace = os.path.dirname(os.getcwd())
sys.path.append(workspace)

from .generator import get_model as Generator
from .discrminator import get_model as Discriminator
from cyclegan.helpers.utils import unet_padding_size, crop
from cyclegan.helpers.plot import plot_heat_map
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def reshape(img):
    img = tf.reshape(img, shape=(1, img.shape[0], img.shape[1], 1))
    return img


inp = np.load('/home/gtzan/data/gan_preprocessing/guitar1/guitar1-046.npy')
tar = np.load('/home/gtzan/data/gan_preprocessing/guitar1/guitar1-049.npy')

PAD_SIZE = ((0, 0), unet_padding_size(inp.shape[1], pool_size=2, layers=8))
IPT_SHAPE = [inp.shape[0], inp.shape[1] + PAD_SIZE[1][0] + PAD_SIZE[1][1]]

pic = np.pad(inp, PAD_SIZE)
pic = np.repeat(pic.reshape((1, IPT_SHAPE[0], IPT_SHAPE[1], 1)), 3, axis=3)

g_model = Generator(IPT_SHAPE[0], IPT_SHAPE[1])
d_model = Discriminator(IPT_SHAPE[0], IPT_SHAPE[1])

g_out = g_model(pic, training=False)

g_pic = tf.reshape(g_out, [g_out.shape[1], g_out.shape[2]])
g_pic = crop(g_pic, PAD_SIZE)
plot_heat_map(g_pic, 'generator')

d_out = d_model([reshape(inp), reshape(g_pic)], training=False)

d_pic = tf.reshape(d_out, [d_out.shape[1], d_out.shape[2]])
plot_heat_map(d_pic, 'discriminator')
