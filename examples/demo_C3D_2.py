''' Original Model/Weights/Sources from https://goo.gl/b8hPpu'''
import os
import cv2
import time
import numpy as np
import argparse
from cv2 import imread
from cv2 import resize
from keras.models import model_from_json


def check_ext(file_name, extensions):
    """Check the file extension

    @param file_name: input file name
    @param extensions: string or list, extension(s) to check

    @return bool: True if it is matched

    """
    if file_name.endswith(extensions):
        return True
    return False


def find_files(dir_path=None, keyword=None, suffix=('.json')):
    """Find files under a directory

    Keyword arguments:
    dir_path -- path of the directory to check (default: '.')
    keyword  -- keyword used to filter files (default: None)
    suffix   -- file extensions to be selected (default: ('.json'))

    @return output: a list of file paths found

    """
    if dir_path is None:
        dir_path = os.getcwd()
    output = []
    for dirPath, dirNames, fileNames in os.walk(dir_path):
        dirmatch = False
        if keyword is not None and dirPath.find(keyword) > 0:
            dirmatch = True
        for f in fileNames:
            if keyword is not None and dirPath.find(keyword) < 0:
                if not dirmatch:
                    continue
            if check_ext(f, suffix):
                output.append(os.path.join(dirPath, f))
    return output


def find_images(dir_path=None, keyword=None):
    """Find images under a directory

    Keyword arguments:
    dir_path -- path of the directory to check (default: '.')
    keyword  -- keyword used to filter images (default: None)

    @return output: a list of images found

    """

    if dir_path is not None and os.path.isfile(dir_path):
        return [dir_path]
    return find_files(dir_path=dir_path, keyword=keyword,
                      suffix=('.jpg', 'png', '.JPEG'))


def read(fimg, size=None):
    """Read image

    @param fimg: input image file name

    Keyword arguments:
    size -- tuple of new size in (height, width)

    """

    img = imread(fimg)
    if img is None:
        print("[IMAGE] Error reading file %s" % fimg)
        return img
    if size is not None:
        img = resize(img, size)

    return img


def prepare_data(img_loc, width, height, sort=False):
    """ Read images as dp inputs

    @param img_loc: path of the images or a list of image paths
    @param width: number rows used to resize the images
    @param height: number columns used to resize the images

    Arguments:
    sort      -- True to sort the images (default: False)

    """

    print('Width = %i, Height = %i' % (width, height))
    if type(img_loc) is list:
        imgs = img_loc
    else:
        imgs = find_images(dir_path=img_loc)
    X = []

    if sort:
        imgs = sorted(imgs)
    for fimg in imgs:
        _img_original = [read(fimg, size=(height, width))]
        if _img_original[0] is None:
            continue
        for c in _img_original:
            X.append(c)

    X = np.array(X).astype('float32')
    return np.array(X)


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
        "--img-width", type=int, default=128, dest='width',
        help="Rows of the images, default: 128."
        )
    parser.add_argument(
        "--img-height", type=int, default=171, dest='height',
        help="Columns of the images, default: 171."
        )

    t0 = time.time()
    args = parser.parse_args()

    path_model = os.path.join(args.ofolder, 'model.json')
    path_weights = os.path.join(args.ofolder, 'weights.h5')
    path_label = os.path.join(args.ofolder, 'labels.txt')

    model = model_from_json(open(path_model).read())
    model.load_weights(path_weights)
    print(model.summary())
    model.compile(loss='mean_squared_error', optimizer='sgd')

    with open(path_label, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))

    X_test = prepare_data(
        args.path, args.width, args.height, sort=True)

    results = []
    detected_lbs = {}
    for i in range(0, X_test.shape[0]-16):
        X = X_test[i:i+16, 8:120, 30:142, :].transpose((3, 0, 1, 2))
        output = model.predict_on_batch(np.array([X]))
        pos_max = output[0].argmax()
        results.append(pos_max)
        if pos_max not in detected_lbs:
            detected_lbs[pos_max] = labels[pos_max]

    print(detected_lbs)

if __name__ == '__main__':
    main()
