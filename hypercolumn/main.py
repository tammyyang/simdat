import os
import cv2
import numpy as np
from keras.optimizers import SGD
from simdat.core import dp_models
from simdat.core import plot
from simdat.core import image

models = dp_models.DPModel()
imnet = dp_models.ImageNet()
im = image.IMAGE()
pl = plot.PLOT()

weight_path = '/tammy/SOURCES/keras/examples/vgg16_weights.h5'
img_path = 'airportwaitingarea_0001.jpg'

model = models.VGG_16(weight_path)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

imgs = im.find_images()
for fimg in imgs:
    print('Processing %s' % fimg)
    name, ext = os.path.splitext(fimg)
    img_original = im.read(fimg, size=(224, 224))
    img = img_original.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    # out = model.predict(img)
    # prob = out.ravel()
    # pl.plot(out.ravel())
    # imagenet_labels_filename = '/tammy/ImageNet/synset_words.txt'
    # results = imnet.find_topk(prob, fname=imagenet_labels_filename)
    # print(results)

    # get_feature = theano.function([model.layers[0].input],
    #                                model.layers[3].get_output(train=False),
    #                                allow_input_downcast=False)
    # feat = get_feature(img)
    # pl.plot_matrix(feat[0][2], fname='feature_l03.png', norm=False,
    #                show_text=False, show_axis=False)

    # get_feature = theano.function([model.layers[13].input],
    #                                model.layers[15].get_output(train=False),
    #                                allow_input_downcast=False)
    # feat = get_feature(img)
    # pl.plot_matrix(feat[0][13], fname='feature_l14.png', norm=False,
    #                show_text=False, show_axis=False)


    layers_extract = [3, 8]
    hc = models.extract_hypercolumn(model, layers_extract, img)
    ave = np.average(hc.transpose(1, 2, 0), axis=2)

    name = name.split('/')[-1]
    pl.plot_matrix(ave, fname='ave_'+name+'.png', norm=False,
                   show_text=False, show_axis=False)

    # cluster = models.cluster_hc(hc)
    # pl.plot_matrix(cluster, fname='cluster.png', norm=False,
    #                show_text=False, show_axis=False)
