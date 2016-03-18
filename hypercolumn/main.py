import os
import cv2
import time
import numpy as np
from keras.optimizers import SGD
from simdat.core import dp_models
from simdat.core import plot
from simdat.core import image

import theano

t0 = time.time()
mdls = dp_models.DPModel()
imnet = dp_models.ImageNet()
im = image.IMAGE()
pl = plot.PLOT()

weight_path = '/home/tammy/SOURCES/keras/examples/vgg16_weights.h5'
img_path = 'images/0001/airportwaitingarea_0004.jpg'
t0 = pl.print_time(t0, 'initiate')

model = mdls.VGG_16(weight_path)
t0 = pl.print_time(t0, 'load weights')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
t0 = pl.print_time(t0, 'compile')

imgs = [img_path]
# imgs = im.find_images()
t0 = pl.print_time(t0, 'find images')
for fimg in imgs:
    t0 = pl.print_time(t0, 'compute for one image')
    print('Processing %s' % fimg)
    name, ext = os.path.splitext(fimg)
    img_original = im.read(fimg, size=(224, 224))
    img = img_original.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    out = model.predict(img)
    prob = out.ravel()
    pl.plot(out.ravel())
    imagenet_labels_filename = '/home/tammy/ImageNet/synset_words.txt'
    results = imnet.find_topk(prob, fname=imagenet_labels_filename)
    print(results)

    '''
    get_feature = theano.function([model.layers[0].input],
                                   model.layers[15].get_output(train=False),
                                   allow_input_downcast=False)
    feat = get_feature(img)
    for i in range(0, 37):
        if i < 10:
            fname = 'feature_l0' + str(i) + '.png'
        else:
            fname = 'feature_l' + str(i) + '.png'
        pl.plot_matrix(feat[0][i], fname=fname, norm=False,
                       show_text=False, show_axis=False)


    # According to the layer structure,
    # ZeroPadding2D (0), Conv.(1), ZP2 (2), Conv(3), MaxPooling2D (4),
    # ZP2(5), Conv.(6), ZP2(7), Conv(8), MP2(9)
    # the first two results from convolutional layers are [3, 8]
    # keep counting, the last two from conv. layers are [22, 29]
    # and the FC layers are [32, 34, 36]

    layers_extract = [3, 8]
    hc = mdls.extract_hypercolumn(model, layers_extract, img)
    ave = np.average(hc.transpose(1, 2, 0), axis=2)

    name = name.split('/')[-1]
    pl.plot_matrix(ave, fname='hc_'+name+'3_8.png', norm=False,
                   show_text=False, show_axis=False)

    layers_extract = [22, 29]
    hc = mdls.extract_hypercolumn(model, layers_extract, img)
    ave = np.average(hc.transpose(1, 2, 0), axis=2)

    name = name.split('/')[-1]
    pl.plot_matrix(ave, fname='hc_'+name+'22_29.png', norm=False,
                   show_text=False, show_axis=False)

    layers_extract = [3, 8, 15, 22, 29]
    hc = mdls.extract_hypercolumn(model, layers_extract, img)
    ave = np.average(hc.transpose(1, 2, 0), axis=2)

    name = name.split('/')[-1]
    pl.plot_matrix(ave, fname='hc_'+name+'8_29_36.png', norm=False,
                   show_text=False, show_axis=False)


    layers_extract = [3, 8, 15, 22, 29]
    hc = mdls.extract_hypercolumn(model, layers_extract, img)
    cluster = mdls.cluster_hc(hc)
    pl.plot_matrix(cluster, fname='hc_cluster.png', norm=False,
                   show_text=False, show_axis=False)
    '''
