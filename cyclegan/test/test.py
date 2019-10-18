import sys
import os

workspace = os.path.dirname(os.getcwd())
sys.path.append(workspace)

from segmentation_models.nestnet import Nestnet
from discrminator import get_model as Discrminator
from gtzan.utils import unet_padding_size, crop, reshape
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


inp = np.load('/home/gtzan/data/gan_preprocessing/guitar1/guitar1-006.npy')
tar = np.load('/home/gtzan/data/gan_preprocessing/guitar1/guitar1-009.npy')

PAD_SIZE = ((0, 0), unet_padding_size(inp.shape[1], pool_size=2, layers=8))
IPT_SHAPE = [inp.shape[0], inp.shape[1] + PAD_SIZE[1][0] + PAD_SIZE[1][1]]

pic = np.pad(inp, PAD_SIZE)
pic = np.repeat(pic.reshape((1, IPT_SHAPE[0], IPT_SHAPE[1], 1)), 3, axis=3)

g_model = Nestnet(backbone_name='vgg16',
                  input_shape=(IPT_SHAPE[0], IPT_SHAPE[1], 3),
                  decoder_filters=(512, 512, 256, 128, 64),
                  decoder_use_batchnorm=True,
                  n_upsample_blocks=5,
                  upsample_rates=(2, 2, 2, 2, 2),
                  activation='tanh')

d_model = Discrminator(IPT_SHAPE[0], IPT_SHAPE[1])

g_out = g_model(pic, training=False)
g_pic = tf.reshape(g_out, [g_out.shape[1], g_out.shape[2]])
g_pic = crop(g_pic, PAD_SIZE)

ax = sns.heatmap(g_pic, robust=True, cbar=True)
ax.invert_yaxis()
plt.savefig('generator.png', format='png', bbox_inches='tight')

d_out = d_model([reshape(inp), reshape(g_pic)],
                training=False)
d_pic = tf.reshape(d_out, [d_out.shape[1], d_out.shape[2]])

plt.close()

ax = sns.heatmap(d_pic, robust=True, cbar=True)
ax.invert_yaxis()
plt.savefig('discrminator.png', format='png', bbox_inches='tight')
