# coding: utf-8

import sys
sys.path.append('/home/gtzan/dc/AT082_23_Orig_Music/cyclegan')
from generator import get_model as Generator
from discrminator import get_model as Discrminator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

inp = np.load('/home/gtzan/data/gan_preprocessing/guitar1/guitar1-006.npy')
tar = np.load('/home/gtzan/data/gan_preprocessing/guitar1/guitar1-009.npy')
IPT_SHAPE = [inp.shape[0], inp.shape[1]]
g_model = Generator(IPT_SHAPE[0], IPT_SHAPE[1])
d_model = Discrminator(IPT_SHAPE[0], IPT_SHAPE[1])

g_out = g_model(inp.reshape(1, IPT_SHAPE[0], IPT_SHAPE[1], 1), training=False)
g_pic = tf.reshape(g_out, [g_out.shape[1], g_out.shape[2]])

ax = sns.heatmap(g_pic, robust=True, cbar=True)
ax.invert_yaxis()
plt.savefig('generator.png', format='png', bbox_inches='tight')

d_out = d_model([inp.reshape(1, IPT_SHAPE[0], IPT_SHAPE[1], 1), g_out],
                training=False)
d_pic = tf.reshape(d_out, [d_out.shape[1], d_out.shape[2]])

plt.close()

ax = sns.heatmap(d_pic, robust=True, cbar=True)
ax.invert_yaxis()
plt.savefig('discrminator.png', format='png', bbox_inches='tight')
