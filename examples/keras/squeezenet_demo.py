"""
Usage: python squeezenet_demo.py -p "/home/db/www/database/tests"
"""
import time
import argparse
from simdat.core import dp_tools
from simdat.core import keras_models as km
from simdat.core import tools
from keras.optimizers import Adam
from keras.optimizers import SGD

dp = dp_tools.DP()
tl = tools.TOOLS()


def main():
    parser = argparse.ArgumentParser(
        description="SqueezeNet example."
        )
    parser.add_argument(
        "--batch-size", type=int, default=32, dest='batchsize',
        help="Size of the mini batch. Default: 32."
        )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of epochs, default 20."
        )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate of SGD, default 0.001."
        )
    parser.add_argument(
        "--epsilon", type=float, default=1e-8,
        help="Epsilon of Adam epsilon, default 1e-8."
        )
    parser.add_argument(
        "-p", "--path", type=str, default='.', required=True,
        help="Path where the images are. Default: $PWD."
        )
    parser.add_argument(
        "--img-width", type=int, default=224, dest='width',
        help="Rows of the images, default: 224."
        )
    parser.add_argument(
        "--img-height", type=int, default=224, dest='height',
        help="Columns of the images, default: 224."
        )
    parser.add_argument(
        "--channels", type=int, default=3,
        help="Channels of the images, default: 3."
        )

    args = parser.parse_args()

    X_train, X_test, Y_train, Y_test, classes = dp.prepare_data_train(
        args.path, args.width, args.height)

    nb_classes = Y_train[0].shape[0]
    print('Total classes are %i' % nb_classes)

    t0 = time.time()
    print "Building the model"
    model = km.SqueezeNet(
        nb_classes, inputs=(args.channels, args.height, args.width))
    dp.visualize_model(model)

    adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=args.epsilon)
    model.compile(
        optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    print "Model built"
    print(model.summary())

    print "Training"
    model.fit(X_train, Y_train, nb_epoch=args.epochs,
              batch_size=args.batchsize, verbose=1)
    print "Model trained"

    print "Evaluating"
    score = model.evaluate(
        X_test, Y_test, batch_size=args.batchsize, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    t0 = tl.print_time(t0, 'score squeezenet')

if __name__ == '__main__':
    main()
