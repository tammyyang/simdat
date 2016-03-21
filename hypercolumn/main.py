import os
import cv2
import time
import numpy as np
from keras.optimizers import SGD
from simdat.core import dp_models
from simdat.core import plot
from simdat.core import image
from simdat.core import ml

t0 = time.time()
mdls = dp_models.DPModel()
imnet = dp_models.ImageNet()
im = image.IMAGE()
pl = plot.PLOT()
mlr = ml.SVMRun()

weight_path = '/tammy/SOURCES/keras/examples/vgg16_weights.h5'
img_path = 'images/0001/airportwaitingarea_0004.jpg'
t0 = pl.print_time(t0, 'initiate')

model = mdls.VGG_16(weight_path)
t0 = pl.print_time(t0, 'load weights')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
t0 = pl.print_time(t0, 'compile')

imgs = im.find_images()
t0 = pl.print_time(t0, 'find images')
X = []
Y = []

for fimg in imgs:
    t0 = pl.print_time(t0, 'compute for one image')
    print('Processing %s' % fimg)
    _cls = int(mlr.get_class_from_path(fimg))
    Y.append(_cls)
    name, ext = os.path.splitext(fimg)
    img_original = im.read(fimg, size=(224, 224))
    img = img_original.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    layers_extract = [3, 8, 15, 22, 29]
    hc = mdls.extract_hypercolumn(model, layers_extract, img)
    # new_shape = 224*224
    # ave = np.average(hc.transpose(1, 2, 0), axis=2).reshape(new_shape)
    # X.append(ave)
    X.append(hc.ravel())

mf = mlr.run(X, Y)
