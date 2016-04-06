import time
import argparse
import numpy as np
from simdat.core import dp_models
from simdat.core import tools
from keras.optimizers import SGD
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
    t0 = tl.print_time(t0, 'prepare data')
    model = mdls.Simple(len(classes), img_row=X_train.shape[2],
                        img_col=X_train.shape[3], colors=X_train.shape[1])
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    t0 = tl.print_time(t0, 'compile the Simple model')

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
