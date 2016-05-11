'''
    Train a simple convnet on the MNIST dataset and visualize filters
    Reference: mnist_cnn.py in keras examples and https://goo.gl/2qIZdC
    This works for Keras 1.0 API
'''
from __future__ import absolute_import
from __future__ import print_function
import os
import warnings
import pylab as pl
import matplotlib.cm as cm
import numpy as np
import numpy.ma as ma
np.random.seed(1337)  # for reproducibility

import theano
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from mpl_toolkits.axes_grid1 import make_axes_locatable


os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'
warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    mosaic = ma.masked_all(
        (nrows * imshape[0] + (nrows - 1) * border,
         ncols * imshape[1] + (ncols - 1) * border), dtype=np.float32)
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


np.set_printoptions(precision=5, suppress=True)
nb_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
i = 4600
pl.imshow(X_train[i, 0], interpolation='nearest', cmap=cm.binary)
print("label : ", Y_train[i, :])

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='valid',
                        input_shape=X_train.shape[1:]))
model.add(Dropout(0.1))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(32, 3, 3))

convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta', metrics=['accuracy'])

WEIGHTS_FNAME = 'mnist_cnn_weights_v1.hdf'
if True and os.path.exists(WEIGHTS_FNAME):
    # Just change the True to false to force re-training
    print('Loading existing weights')
    model.load_weights(WEIGHTS_FNAME)
else:
    batch_size = 128
    nb_epoch = 12
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    model.save_weights(WEIGHTS_FNAME, overwrite=True)

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# K.learning_phase() is a flag that indicates if the network is
# in training or predict phase. It allow layer (e.g. Dropout)
# to only be applied during training
inputs = [K.learning_phase()] + model.inputs
_convout1_f = K.function(inputs, [convout1.output])

# Visualize the first layer of convolutions on an input image
i = 4600
X = X_test[i:i+1]

pl.figure()
pl.title('input')
nice_imshow(pl.gca(), np.squeeze(X), vmin=0, vmax=1,
            cmap=cm.binary, name='mnist_vis_input.png')

# Visualize weights
W = model.layers[0].W.get_value(borrow=True)
W = np.squeeze(W)
print("W shape : ", W.shape)
pl.figure(figsize=(15, 15))
pl.title('conv1 weights')
nice_imshow(pl.gca(), make_mosaic(W, 6, 6), cmap=cm.binary,
            name='mnist_vis_conv1_weights.png')


# Visualize convolution result (after activation)
def convout1_f(X):
    # The [0] is to disable the training phase flag
    return _convout1_f([0] + [X])


C1 = convout1_f(X)
C1 = np.squeeze(C1)
print("C1 shape : ", C1.shape)

pl.figure(figsize=(15, 15))
pl.suptitle('convout1')
nice_imshow(pl.gca(), make_mosaic(C1, 6, 6), cmap=cm.binary,
            name='mnist_vis_convout1.png')
