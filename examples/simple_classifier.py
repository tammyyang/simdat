import numpy as np
import argparse
from simdat.core import dp_models
from simdat.core import image
from simdat.core import ml
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils

im = image.IMAGE()
mlr = ml.MLRun()


def prepare_data(path, img_rows, img_cols):
    imgs = im.find_images(dir_path=path)
    X = []
    Y = []
    classes = {}
    for fimg in imgs:
        _cls_ix = mlr.get_class_from_path(fimg)
        if _cls_ix not in classes:
            classes[_cls_ix] = len(classes)
        Y.append(classes[_cls_ix])
        _img_original = im.read(fimg, size=(img_rows, img_cols))
        _img = _img_original.transpose((2, 0, 1))
        X.append(_img)

    nb_classes = len(classes)

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, y_train, y_test = mlr.split_samples(X, Y)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return X_train, X_test, Y_train, Y_test, classes


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

    args = parser.parse_args()
    np.random.seed(args.seed)

    X_train, X_test, Y_train, Y_test, classes = prepare_data(
        args.path, args.rows, args.cols)
    mdls = dp_models.DPModel()
    model = mdls.Simple(len(classes), img_row=X_train.shape[2],
                        img_col=X_train.shape[3], colors=X_train.shape[1])
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    model.fit(X_train, Y_train, batch_size=args.batchsize,
              nb_epoch=args.epochs, show_accuracy=True, verbose=1,
              validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    main()
