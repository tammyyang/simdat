'''Visualization of the convolutional filters of VGG16

References:
    https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

Before running this script, download the weights for the VGG16 model at:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
and make sure the variable `weights_path` in this script matches the location.
'''
from __future__ import print_function
import numpy as np
import numpy.ma as ma
import time
import os
import math
import pylab as pl
import matplotlib.cm as cm
from keras import backend as K
from mpl_toolkits.axes_grid1 import make_axes_locatable
from simdat.core import dp_models
from simdat.core import tools


def nice_imshow(ax, data, name='output.png', vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax,
                   interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)
    pl.savefig(name)


def make_mosaic(imgs, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    nrows = math.ceil(math.sqrt(nimgs))
    imshape = imgs.shape[1:]
    mosaic = ma.masked_all(
        (nrows * imshape[0] + (nrows - 1) * border,
         nrows * imshape[1] + (nrows - 1) * border), dtype=np.float32)
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / nrows))
        col = i % nrows
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


# Visualize convolution result (after activation)
def conv_f(X):
    # The [0] is to disable the training phase flag
    return _conv_f([0] + [X])

# simdat dependencies
dp = dp_models.DPModel()
tl = tools.DATA()

# basic parameters
img_width = 224
img_height = 224
weights_path = '/home/tammy/www/vgg-16/vgg16_weights.h5'
img_path = '/home/tammy/www/database/food-test/broccoli/broccoli.jpg'

model = dp.VGG_16(weights_path=weights_path)
model.summary()
X, Y, cls, F = dp.prepare_data_test(
    img_path, img_width, img_height, convert_Y=False, y_as_str=False)
inputs = [K.learning_phase()] + model.inputs

for layer in model.layers:
    lname = dp.is_convolutional(layer)
    if lname is None:
        continue
    # return the output of a certain layer given a certain input
    # http://keras.io/getting-started/faq/
    _conv_f = K.function(inputs, [layer.output])
    C1 = conv_f(X)
    C1 = np.squeeze(C1)
    print("%s shape : " % lname, C1.shape)
    pl.figure(figsize=(15, 15))
    pl.suptitle(lname)
    nice_imshow(pl.gca(), make_mosaic(C1), cmap=cm.binary,
                name=lname + '.png')

# TODO: Visualize weights
