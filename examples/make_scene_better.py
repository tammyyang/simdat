import cv2
import os
import sys
import argparse
import logging
import numpy as np
from simdat.core import image
from simdat.core import plot

imgtl = image.OverlayTextDetection()


def test(imgs):
    pl = plot.PLOT()
    for _img in imgs:
        print('Processing %s' % _img)
        img = imgtl.read(_img)
        lmb = imgtl.linked_map_boundary(img, output=True)
        lmb = lmb.astype('uint8')
        contours = imgtl.contours(lmb)
        amin = img.shape[0]*img.shape[1]*0.01
        amax = img.shape[0]*img.shape[1]*0.2
        imgtl.draw_contours(img, contours, amin=amin, amax=amax, output=True)

        # a = np.bincount(index[0])
        # b = np.bincount(index[1])
        # pl.plot(a, fname='a.png')
        # pl.plot(b, fname='b.png')
        # for i in range(0, len(index[0])):
        #     print index[0][i], index[1][i]
        # lines = imgtl.get_houghlines(lmb)
        # imgtl.draw_houghlines(img, lines, output=True)


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
                help="Select action: black-bar/detect-text \
                      (default: black-bar)."
                )
    parser.add_argument(
                "-v", "--verbosity", action="count",
                help="increase output verbosity"
                )
    parser.add_argument(
                "-t", "--test", action='store_true'
                )
    args = parser.parse_args()

    log_level = logging.WARNING
    if args.verbosity == 1:
        log_level = logging.INFO
    elif args.verbosity == 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level,
                        format='[MSB %(levelname)s] %(message)s')

    if args.fname is not None:
        imgtl.check_exist(args.fname)
        imgs = [args.fname]
    elif args.dir is not None:
        imgs = imgtl.find_images(dir_path=args.dir)
    else:
        imgs = imgtl.find_images()

    if args.test:
        test(imgs)
        sys.exit(1)

    for fimg in imgs:
        name, ext = os.path.splitext(fimg)
        img = imgtl.read(fimg)
        gray = imgtl.gray(img)
        if args.action == 'black-bar':
            fname = ''.join([name, '_crop', ext])
            imgtl.crop_black_bars(img, fname=fname)
        elif args.action == 'detect-text':
            fname = ''.join([name, '_contour', ext])
            contours = imgtl.detect_text_area(gray, save=True)
            imgtl.draw_contours(img, contours, fname=fname)

if __name__ == '__main__':
    main()
