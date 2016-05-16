"""
Usage: python squeezenet_demo.py -p "/home/db/www/database/tests"
"""
import argparse
from simdat.core import dp_models
from keras.optimizers import Adam

dp = dp_models.DPModel()


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
        "--img-width", type=int, default=227, dest='width',
        help="Rows of the images, default: 227."
        )
    parser.add_argument(
        "--img-height", type=int, default=227, dest='height',
        help="Columns of the images, default: 227."
        )
    parser.add_argument(
        "--channels", type=int, default=3,
        help="Channels of the images, default: 3."
        )

    args = parser.parse_args()

    X_train, X_test, Y_train, Y_test, classes = dp.prepare_data_train(
        args.path, args.width, args.height)

    nb_classes = Y_train[0].shape[0]

    print "Building the model"
    graph = dp.SqueezeNet(
        nb_classes, inputs=(args.channels, args.height, args.width))
    dp.visualize_model(graph)

    adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=args.epsilon)
    graph.compile(optimizer=adam, loss='categorical_crossentropy')
    print "Model built"

    print "Training"
    graph.fit({'input': X_train, 'output': Y_train},
              batch_size=args.batchsize, nb_epoch=args.epochs,
              validation_split=0.1, verbose=1)
    print "Model trained"

    print "Evaluating"
    score = graph.evaluate({'input': X_test, 'output': Y_test},
                           batch_size=args.batchsize, verbose=1)
    print score

if __name__ == '__main__':
    main()
