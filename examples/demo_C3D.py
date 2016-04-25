''' Original Model/Weights/Sources from https://goo.gl/b8hPpu'''
import os
import cv2
import numpy as np
import argparse
from simdat.core import dp_models
from simdat.core import tools
from keras.models import model_from_json


def main():
    parser = argparse.ArgumentParser(
        description="Demo Sports 1M C3D Network on Keras"
        )
    parser.add_argument(
        "--model-loc", type=str, default=os.getcwd(), dest='ofolder',
        help="Path of the folder to output or to load the model."
        )
    parser.add_argument(
        "-p", "--path", type=str, default=None, required=True,
        help="Path of the video."
        )
    parser.add_argument(
        "--img-rows", type=int, default=244, dest='rows',
        help="Rows of the images, default: 171."
        )
    parser.add_argument(
        "--img-cols", type=int, default=244, dest='cols',
        help="Columns of the images, default: 128."
        )

    t0 = time.time()
    mdls = dp_models.DPModel()
    tl = tools.TOOLS()
    args = parser.parse_args()
    path_model = os.path.join(args.ofolder, 'model.json')
    path_weights = os.path.join(args.ofolder, 'weights.h5')
    path_label = os.path.join(args.ofolder, 'labels.txt')
    t0 = tl.print_time(t0, 'init')

    model = model_from_json(open(path_model).read())
    model.load_weights(path_weights)
    print(model.summary())
    t0 = tl.print_time(t0, 'load model')
    model.compile(loss='mean_squared_error', optimizer='sgd')
    t0 = tl.print_time(t0, 'compile model')

    with open(path_label, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))

    X_test, Y_test, classes, F = mdls.prepare_data_test(
        args.path, args.rows, args.cols)
    t0 = tl.print_time(t0, 'load data')
    # c x l x h x w where c is the number of
    # channels, l is length in number of frames, h and w are the
    # height and width of the frame
    # Original shape = (16, 3, 122, 122)
    # New shape = (3, 16, 122, 122)
    for i in range(0, X_test.shape[0]-17):
        X = X_test[i:i+16, :, 8:120, 30:142].transpose((1, 0, 2, 3))
        output = model.predict_on_batch(np.array([X]))

        print('==============')
        print('Frame %i to %i' % (i, i+16))
        print('Position of maximum probability: {}'.format(output[0].argmax()))
        print('Maximum probability: {:.5f}'.format(max(output[0][0])))
        print('Corresponding label: {}'.format(labels[output[0].argmax()]))
    t0 = tl.print_time(t0, 'predict')

if __name__ == '__main__':
    main()
