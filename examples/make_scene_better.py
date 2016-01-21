import cv2
import os
import sys
import argparse
import logging
import numpy as np
from scipy import ndimage
from simdat.core import image
from simdat.core import plot

imgtl = image.OverlayTextDetection()


def test(imgs):
    pl = plot.PLOT()
    for _img in imgs:
        print('Processing %s' % _img)
        img = imgtl.read(_img)
        lmb = imgtl.linked_map_boundary(img, output=False)
        ''' Use LBP '''
        lbp = imgtl.LBP(lmb, subtract=True)
        thre = np.amax(lbp)*0.3
        lbp = imgtl.select(lbp, thre*0.5, thre)
        pl.plot_matrix(lbp, show_text=False, norm=False, fname='lbp.png')

        '''
        mean = np.mean(lbp, axis=1)
        pl.plot(mean, fname='mean_row.png')
        sigma = mean.mean() + 2*mean.std()
        greater = [i for i in range(0, len(mean)) if (mean[i] > sigma)]
        thre = img.shape[0]*0.2
        mappingy = np.zeros(mean.shape)
        for i in range(0, len(greater)-1):
            s = greater[i]
            e = greater[i+1]
            if e - s < thre:
                mappingy[s:e+1] = 1
        mappingy = mappingy.reshape((mappingy.shape[0], 1))
        mappingy = np.repeat(mappingy, img.shape[1], axis=1)
        pl.plot_matrix(mappingy, show_text=False, norm=False, fname='mappingy.png')

        mean = np.mean(lbp, axis=0)
        pl.plot(mean, fname='mean_col.png')
        sigma = mean.mean() + 2*mean.std()
        greater = [i for i in range(0, len(mean)) if (mean[i] > sigma)]
        thre = int(img.shape[1]*0.2)
        mappingx = np.zeros(mean.shape)
        for i in range(0, len(greater)-1):
            s = greater[i]
            e = greater[i+1]
            if e - s < thre:
                mappingx[s:e+1] = 1
        mappingx = mappingx.reshape((mappingx.shape[0], 1))
        mappingx = mappingx.T
        mappingx = np.repeat(mappingx, img.shape[0], axis=0)
        pl.plot_matrix(mappingx, show_text=False, norm=False, fname='mappingx.png')
        mapping = mappingx + mappingy
        pl.plot_matrix(mapping, show_text=False, norm=False, fname='mapping.png')

        # res = mapping*lmb
        res = mapping*lbp
        '''
        res = lbp
        pl.plot_matrix(res, show_text=False, norm=False, fname='res.png')
        res = res.astype('uint8')
        h = int(img.shape[0]*0.1)
        w = int(img.shape[1]*0.20)
        kernel = np.ones((h, w),np.uint8)
        res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
        pl.plot_matrix(res, show_text=False, norm=False, fname='closing.png')

        contours = imgtl.contours(res)
        amin = img.shape[0]*img.shape[1]*0.01
        amax = img.shape[0]*img.shape[1]*0.5
        imgtl.draw_contours(img, contours, amin=amin, amax=amax, output=True)
        imgtl.save(img, 'contours.png')

        ''' Use houghlines '''
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
