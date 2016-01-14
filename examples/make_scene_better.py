import cv2
import os
import argparse
import logging
import numpy as np
from simdat.core import tools


def crop_black_bars(fimg, thre=1, save=True):
    name, ext = os.path.splitext(fimg)
    img = cv2.imread(fimg)

    _gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _mean = np.array(_gray).mean(axis=1)
    _selected = [i for i in range(0, len(_mean)) if _mean[i] > thre]

    start = _selected[0]
    end = _selected[-1]

    logging.debug(_mean[100:])
    logging.debug(_mean[:100])
    logging.info('Selected image raws line %i to line %i' % (start, end))

    if save:
        fname = ''.join([name, '_crop', ext])
        logging.info('Saving croped file as %s.' % fname)
        cv2.imwrite(fname, img[start:end])


def main():

    parser = argparse.ArgumentParser(
                description="Simple tool to make the scene image better."
                )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
                "-f", "--fname", type=str, default=None,
                help="Specify the filename of the image to handle."
                )
    group.add_argument(
                "-d", "--dir", type=str, default=None,
                help="Specify the directory to look for images (default .)."
                )
    parser.add_argument(
                "-a", "--action", type=str, default='black-bar',
                help="Select action: black-bar/ (default: black-bar)."
                )
    parser.add_argument(
                "-v", "--verbosity", action="count",
                help="increase output verbosity"
                )
    args = parser.parse_args()

    log_level = logging.WARNING
    if args.verbosity == 1:
        log_level = logging.INFO
    elif args.verbosity == 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level,
                        format='[MSB %(levelname)s] %(message)s')

    imgtl = tools.IMAGE()
    if args.fname is not None:
        imgtl.check_exist(args.fname)
        imgs = [args.fname]
    elif args.dir is not None:
        imgs = imgtl.find_images(dir_path=args.dir)
    else:
        imgs = imgtl.find_images()

    for fimg in imgs:
        if args.action == 'black-bar':
            imgtl.crop_black_bars(fimg)


if __name__ == '__main__':
    main()
