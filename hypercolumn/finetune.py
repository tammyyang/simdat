import os
import cv2
import time
import numpy as np
from keras.optimizers import SGD
from simdat.core import dp_models
from simdat.core import ml
from simdat.core import plot
from simdat.core import image

import theano

im = image.IMAGE()
pl = plot.PLOT()
mlr = ml.MLRun()

t0 = time.time()
mdls = dp_models.DPModel()
imnet = dp_models.ImageNet()

weight_path = '/tammy/SOURCES/keras/examples/vgg16_weights.h5'
t0 = pl.print_time(t0, 'initiate')

model = mdls.VGG_16(weight_path)
t0 = pl.print_time(t0, 'load weights')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
t0 = pl.print_time(t0, 'compile')

imgs = im.find_images()
X = []
Y = []

for fimg in imgs:
    Y.append(int(mlr.get_class_from_path(fimg)))
    X.append(im.read(fimg, size=(224, 224)))

X = np.array(X)
Y = np.array(Y)
print(X.shape)

train_X, test_X, train_Y, test_Y = mlr.split_samples(X, Y)

train_X = train_X.reshape(train_X.shape[0], 3, 224, 224)
test_X = test_X.reshape(test_X.shape[0], 3, 224, 224)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

train_X /= 225
test_X /= 225


for stack in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
    for l in mdls.layers[stack]:
        l.trainable = False

batch_size = 128
nb_epoch = 5

model.fit(train_X, train_Y,
          batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1,
          validation_data=(test_X, test_Y))
score = model.evaluate(test_X, test_Y, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

