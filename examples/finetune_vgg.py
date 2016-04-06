import time
import argparse
import numpy as np
from simdat.core import dp_models
from keras.optimizers import SGD
from simdat.core import tools
from keras.layers.core import Dense, Activation
from keras.utils import np_utils


def main():
    parser = argparse.ArgumentParser(
        description="Use Simple model to train a classifier."
        )
    parser.add_argument(
        "-p", "--path", type=str, default='.',
        help="Path where the images are. Default: $PWD."
        )
    parser.add_argument(
        "-w", "--weights", type=str,
        default='/home/tammy/SOURCES/keras/examples/vgg16_weights.h5',
        help="Path of vgg weights"
        )
    parser.add_argument(
        "--img-rows", type=int, default=224, dest='rows',
        help="Rows of the images, default: 224."
        )
    parser.add_argument(
        "--img-cols", type=int, default=224, dest='cols',
        help="Columns of the images, default: 224."
        )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="Random seed, default: 1337."
        )
    parser.add_argument(
        "--batch-size", type=int, default=64, dest='batchsize',
        help="Size of the mini batch. Default: 64."
        )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of epochs, default 20."
        )

    t0 = time.time()
    mdls = dp_models.DPModel()
    tl = tools.TOOLS()

    args = parser.parse_args()
    np.random.seed(args.seed)

    X_train, X_test, Y_train, Y_test, classes = mdls.prepare_data(
        args.path, args.rows, args.cols)
    nclasses = len(classes)
    t0 = tl.print_time(t0, 'prepare data')

    model = mdls.VGG_16(args.weights, lastFC=False)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    print('[finetune_vgg] Adding Dense(nclasses) and Activation(softmax)')
    model.add(Dense(nclasses, activation='softmax'))
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    t0 = tl.print_time(t0, 'compile the model to be fine tuned.')

    for stack in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
        for l in mdls.layers[stack]:
            l.trainable = False

    model.fit(X_train, Y_train, batch_size=args.batchsize,
              nb_epoch=args.epochs, show_accuracy=True, verbose=1,
              validation_data=(X_test, Y_test))
    t0 = tl.print_time(t0, 'fit')
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    t0 = tl.print_time(t0, 'evaluate')

if __name__ == '__main__':
    main()
