import os
import sys
import h5py
import numpy as np
import scipy as sp
from collections import OrderedDict
from sklearn import cluster
from simdat.core import image
from simdat.core import ml
from keras import regularizers
from keras.models import Sequential
# from keras.models import Graph
from keras.models import Model
from keras.layers import Input, Activation, merge
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.utils import np_utils


class DP:
    def __init__(self):
        self.im = image.IMAGE()
        self.mlr = ml.MLRun()
        self.dp_init()

    def dp_init(self):
        """ place holder for child class """
        pass

    def prepare_cifar10_data(self, nb_classes=10):
        """ Get Cifar10 data """

        from keras.datasets import cifar10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        return X_train, X_test, Y_train, Y_test

    def prepare_mnist_data(self, rows=28, cols=28, nb_classes=10):
        """ Get MNIST data """

        from keras.datasets import mnist

        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(X_train.shape[0], 1, rows, cols)
        X_test = X_test.reshape(X_test.shape[0], 1, rows, cols)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        return X_train, X_test, Y_train, Y_test

    def visualize_model(self, model, to_file='model.png'):
        '''Visualize model (work with Keras 1.0)'''

        if type(model) == Sequential or type(model) == Model:
            from keras.utils.visualize_util import plot
            plot(model, to_file=to_file)

        elif type(model) == Graph:
            import pydot
            graph = pydot.Dot(graph_type='digraph')
            config = model.get_config()
            for input_node in config['input_config']:
                graph.add_node(pydot.Node(input_node['name']))

            for layer_config in [config['node_config'],
                                 config['output_config']]:
                for node in layer_config:
                    graph.add_node(pydot.Node(node['name']))
                    if node['inputs']:
                        for e in node['inputs']:
                            graph.add_edge(pydot.Edge(e, node['name']))
                    else:
                        graph.add_edge(pydot.Edge(node['input'], node['name']))

            graph.write_png(to_file)

    def extract_hypercolumn(self, model, la_idx, instance):
        ''' Extract HyperColumn of pixels (Theano Only)

        @param model: input DP model
        @param la_idx: indexes of the layers to be extract
        @param instamce: image instance used to extract the hypercolumns

        '''
        import theano
        layers = [model.layers[li].get_output(train=False) for li in la_idx]
        get_feature = theano.function([model.layers[0].input], layers,
                                      allow_input_downcast=False)
        feature_maps = get_feature(instance)
        hypercolumns = []
        for convmap in feature_maps:
            for fmap in convmap[0]:
                upscaled = sp.misc.imresize(fmap, size=(224, 224),
                                            mode="F", interp='bilinear')
                hypercolumns.append(upscaled)
        return np.asarray(hypercolumns)

    def is_dense(self, layer):
        '''Check if the layer is dense (fully connected)
           Return layer name if it is a dense layer, None otherwise'''

        layer_name = layer.get_config()['name']
        ltype = layer_name.split('_')[0]
        if ltype == 'dense':
            return layer_name
        return None

    def is_convolutional(self, layer):
        '''Check if the layer is convolutional
           Return layer name if it is a dense layer, None otherwise'''

        layer_name = layer.get_config()['name']
        ltype = layer_name.split('_')[0]
        if ltype.find('convolution') > -1:
            return layer_name
        return None

    def cluster_hc(self, hc, n_jobs=1):
        ''' Use KMeans to cluster hypercolumns'''

        ori_size = hc.shape[1]
        new_size = ori_size*ori_size
        m = hc.transpose(1, 2, 0).reshape(new_size, -1)
        kmeans = cluster.KMeans(n_clusters=2, max_iter=300, n_jobs=n_jobs,
                                precompute_distances=True)
        cluster_labels = kmeans .fit_predict(m)
        imcluster = np.zeros((ori_size, ori_size))
        imcluster = imcluster.reshape((new_size,))
        imcluster = cluster_labels
        return imcluster.reshape(ori_size, ori_size)

    def prepare_data(self, img_loc, width, height, convert_Y=True,
                     rc=False, scale=True, classes=None,
                     sort=False, trans=True):
        """ Read images as dp inputs

        @param img_loc: path of the images or a list of image paths
        @param width: number rows used to resize the images
        @param height: number columns used to resize the images

        Arguments:
        rc        -- True to random crop the images as four (default: False)
        scale     -- True to divide input images by 255 (default: True)
        classes   -- A pre-defined list of class index (default: None)
        convert_Y -- True to use np_utils.to_categorical to convert Y
                     (default: True)
        sort      -- True to sort the images (default: False)
        trans     -- True to transport the image from (h, w, c) to (c, h, w)

        """

        print('[dp_models] width = %i, height = %i' % (width, height))
        if type(img_loc) is list:
            imgs = img_loc
        else:
            imgs = self.im.find_images(dir_path=img_loc)
        X = []
        Y = []
        F = []
        create_new_cls = False
        if classes is None:
            create_new_cls = True
            classes = []
        counter = 0

        if rc:
            print('[DP] Applying random crop to the image')
        if sort:
            imgs = sorted(imgs)
        for fimg in imgs:
            if counter % 1000 == 0:
                print('[DP] Reading images: %i' % counter)
            _cls_ix = self.mlr.get_class_from_path(fimg)
            if _cls_ix not in classes and create_new_cls:
                classes.append(_cls_ix)

            if rc:
                _img_original = self.im.read_and_random_crop(
                    fimg, size=(height, width)).values()
            else:
                _img_original = [self.im.read(fimg, size=(height, width))]

            if _img_original[0] is None:
                continue
            for c in _img_original:
                if trans:
                    img = c.transpose((2, 0, 1))
                else:
                    img = c
                X.append(img)
                Y.append(classes.index(_cls_ix))
                F.append(os.path.basename(fimg))
            counter += 1
            fname = str(counter) + '.jpg'

        X = np.array(X).astype('float32')
        if scale:
            X /= 255
        if convert_Y:
            Y = np_utils.to_categorical(np.array(Y), len(classes))

        return np.array(X), np.array(Y), classes, F

    def prepare_data_test(self, img_loc, width, height,
                          convert_Y=True, trans=True,
                          scale=True, classes=None, y_as_str=True):
        """ Read images as dp inputs

        @param img_loc: path of the images or a list of image paths
        @param width: number rows used to resize the images
        @param height: number columns used to resize the images

        Arguments:
        y_as_str  -- True to return Y as a list of class strings
                     This overwrites convert_Y as False. (default: True)
        convert_Y -- True to use np_utils.to_categorical to convert Y
                     (default: True)

        """
        if y_as_str:
            X, Y, classes, F = self.prepare_data(
                img_loc, width, height, sort=True, trans=trans,
                scale=scale, classes=classes, convert_Y=False)
            _Y = [classes[_y] for _y in Y]
            return X, _Y, classes, F
        X, Y, classes, F = self.prepare_data(
            img_loc, width, height, scale=scale, trans=trans,
            classes=classes, convert_Y=convert_Y, sort=True)
        return X, Y, classes, F

    def prepare_data_train(self, img_loc, width, height, sort=False,
                           trans=True, test_size=None, rc=False,
                           scale=True, classes=None):
        """ Read images as dp inputs

        @param img_loc: path of the images or a list of image paths
        @param width: number rows used to resize the images
        @param height: number columns used to resize the images

        Arguments:

        sort      -- True to sort the images (default: False)
        test_size -- size of the testing sample (default: 0.33)

        """

        X, Y, classes, F = self.prepare_data(
            img_loc, width, height, rc=rc, trans=trans,
            scale=scale, classes=classes, sort=sort)

        if type(test_size) is float:
            self.mlr.args.test_size = test_size
            print('[DP] Changing test_size to %f. The one written in'
                  'ml.json will be overwritten!' % test_size)

        X_train, X_test, Y_train, Y_test = self.mlr.split_samples(X, Y)
        print('[DP] X_train shape: (%i, %i)'
              % (X_train.shape[0], X_train.shape[1]))
        print('[DP] Y_train shape: (%i, %i)'
              % (Y_train.shape[0], Y_train.shape[1]))
        print('[DP] %i train samples' % X_train.shape[0])
        print('[DP] %i test samples' % X_test.shape[0])

        return X_train, X_test, Y_train, Y_test, classes


class DPModel(DP):
    def dp_init(self):
        """ init called by the DP class """
        self.layers = None
        self.dpmodel_init()

    def dpmodel_init(self):
        """ place holder for child class """
        pass

    def VGG_16(self, weights_path=None, lastFC=True):
        '''VGG-16 model, source from https://goo.gl/qqM88H'''

        self.layers = OrderedDict([
            ('conv1', [
                ZeroPadding2D((1, 1), input_shape=(3, 224, 224)),
                Convolution2D(64, 3, 3, activation='relu'),
                ZeroPadding2D((1, 1)),
                Convolution2D(64, 3, 3, activation='relu'),
                MaxPooling2D((2, 2),  strides=(2, 2))
            ]),
            ('conv2', [
                ZeroPadding2D((1, 1)),
                Convolution2D(128, 3, 3, activation='relu'),
                ZeroPadding2D((1, 1)),
                Convolution2D(128, 3, 3, activation='relu'),
                MaxPooling2D((2, 2), strides=(2, 2))
            ]),
            ('conv3', [
                ZeroPadding2D((1, 1)),
                Convolution2D(256, 3, 3, activation='relu'),
                ZeroPadding2D((1, 1)),
                Convolution2D(256, 3, 3, activation='relu'),
                ZeroPadding2D((1, 1)),
                Convolution2D(256, 3, 3, activation='relu'),
                MaxPooling2D((2, 2), strides=(2, 2))
            ]),
            ('conv4', [
                ZeroPadding2D((1, 1)),
                Convolution2D(512, 3, 3, activation='relu'),
                ZeroPadding2D((1, 1)),
                Convolution2D(512, 3, 3, activation='relu'),
                ZeroPadding2D((1, 1)),
                Convolution2D(512, 3, 3, activation='relu'),
                MaxPooling2D((2, 2), strides=(2, 2))
            ]),
            ('conv5', [
                ZeroPadding2D((1, 1)),
                Convolution2D(512, 3, 3, activation='relu'),
                ZeroPadding2D((1, 1)),
                Convolution2D(512, 3, 3, activation='relu'),
                ZeroPadding2D((1, 1)),
                Convolution2D(512, 3, 3, activation='relu'),
                MaxPooling2D((2, 2), strides=(2, 2))
            ]),
            ('fc', [
                Flatten(),
                Dense(4096, activation='relu'),
                Dropout(0.5),
                Dense(4096, activation='relu'),
                Dropout(0.5),
            ]),
            ('classify', [
                Dense(1000, activation='softmax')
            ])
        ])
        model = Sequential()
        for stack in self.layers:
            if stack == 'classify' and not lastFC:
                print('[DPModels] Skip the last FC layer')
                continue
            for l in self.layers[stack]:
                model.add(l)

        if weights_path:
            self.load_weights(model, weights_path, lastFC=lastFC)

        return model

    def load_weights(self, model, weights_path, lastFC=True):
        """ Load model weights

        @param model: the model
        @param weight_path: path of the weight file

        Arguments:

        lastFC -- True to load weights for the last FC layer (default: True)

        """
        if lastFC:
            model.load_weights(weights_path)
            return
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        return

    def Simple(self, cats, img_row=224, img_col=224, conv_size=3,
               colors=3, weights_path=None, filter_size=32):
        '''Simple conv model to the best MNIST result.
           The input shape should be (n_imgs, colors, img_row, img_col).

        @param cats: total number of final categories

        Arguments:
        img_row -- number of rows of the image
        img_col -- number of cols of the image
        conv_size -- size of the conv window
        colors  -- colors of the image (3 for RGB)
        weights_path -- used for loading the existing model
        filter_size -- number of the filters of the first conv layer

        '''
        self.layers = OrderedDict([
            ('conv1', [
                Convolution2D(filter_size, conv_size, conv_size,
                              activation='relu', border_mode='valid',
                              input_shape=(colors, img_row, img_col)),
                Convolution2D(filter_size*2, conv_size, conv_size,
                              activation='relu'),
                MaxPooling2D((2, 2),  strides=(1, 1)),
                Dropout(0.25)
            ]),
            ('fc', [
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
            ]),
            ('classify', [
                Dense(cats, activation='softmax')
            ])
        ])
        model = Sequential()
        for stack in self.layers:
            for l in self.layers[stack]:
                model.add(l)

        if weights_path:
            model.load_weights(weights_path)

        return model


class ImageNet(image.IMAGE):
    def get_labels(self, fname='synset_words.txt'):
        '''Get ImageNet labels from file'''

        if not self.check_exist(fname):
            print('ERROR: Cannot find %s.' % fname)
            sys.exit(1)
        return np.loadtxt(fname, str, delimiter='\t')

    def find_topk(self, prob, labels=None, fname='synset_words.txt', ntop=3):
        ''' Find the categories with highest probabilities

        @param prob: a list of probabilities of 1,000 categories

        Keyword arguments:
        labels -- ImageNet labels (default: re-gain from self.get_labels)
        fname  -- filename of the ImageNet labels
        ntop   -- how many top cats to be shown (default: 3)

        @retuen results = {INDEX: (probability, label, [cats])}
         example: {955: (0.00051713100401684642, 'n07754684',
                   ['jackfruit', 'jak', 'jack'])}

        '''
        if labels is None:
            labels = self.get_labels(fname)
        results = {}
        top_k = prob.flatten().argsort()[-1:-(ntop+1):-1]
        for k in top_k:
            l = labels[k].replace(',', '').split(' ')
            results[k] = (prob[k], l[0], l[1:])
        return results
