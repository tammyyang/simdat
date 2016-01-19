import os
import subprocess
import numpy as np
import logging
from PIL import Image
from simdat.core import tools
from cv2 import imwrite
from cv2 import Sobel, CV_64F, Laplacian
from cv2 import cvtColor, COLOR_BGR2GRAY
from cv2 import GaussianBlur
from cv2 import adaptiveThreshold
from cv2 import morphologyEx, MORPH_CLOSE
from cv2 import boundingRect, rectangle
from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_NONE


class IMAGE(tools.TOOLS):
    def find_images(self, dir_path=None, keyword=None):
        """Find images under a directory


        Keyword arguments:
        dir_path -- path of the directory to check (default: '.')
        keyword  -- keyword used to filter images (default: None)

        @return output: a list of images found

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
                if self.check_ext(f, ('.jpg', 'png')):
                    output.append(os.path.join(dirPath, f))
        return output

    def get_img_info(self, img_path):
        """Find image size and pixel array

        @param img_path: path of the input image

        @return image.size: tuple, size of the image
        @return pix: pixel of the image

        """
        im = Image.open(img_path)
        pix = im.load()
        return im.size, pix

    def get_images(self, path):
        """Find images from the given path"""

        if os.path.isfile(path):
            if self.check_ext(path, ('.jpg', 'png')):
                return [path]
        elif os.path.isdir(path):
            return self.find_images(path)

    def crop_black_bars(self, img, fname=None, thre=1):
        """Crop symmetric black bars"""

        _gray = cvtColor(img, COLOR_BGR2GRAY)
        _mean = np.array(_gray).mean(axis=1)
        _selected = [i for i in range(0, len(_mean)) if _mean[i] > thre]
        _start = _selected[0]
        _end = _selected[-1]
        cut = max(_start, len(img) - _end)

        logging.debug(_mean[100:])
        logging.debug(_mean[:100])
        logging.info('Cut %i lines' % cut)

        if fname is not None:
            logging.info('Saving croped file as %s.' % fname)
            self.save(img[cut:-cut], fname)

    def laplacian(self, img, fname=None):
        """Laplacian transformation"""

        la = Laplacian(img, CV_64F)
        if fname is not None:
            self.save(la, fname)
        return la

    def sobel(self, img, axis=0):
        """Sobel transformation"""

        if axis == 0:
            sobel = Sobel(img, CV_64F, 0, 1, ksize=5)
        elif axis == 1:
            sobel = Sobel(img, CV_64F, 1, 0, ksize=5)
        if fname is not None:
            self.save(sobel, fname)
        return sobel

    def detect_text_area(self, img, save=False):
        """Detect text area using simple opencv methods"""

        blur = GaussianBlur(img, (5, 5), 0)
        thresh = adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        se = np.ones((7, 7), dtype='uint8')
        if save:
            self.save(blur, 'blur.png')
            self.save(thresh, 'thresh.png')
        contours, hierarchy = findContours(thresh, RETR_EXTERNAL,
                                           CHAIN_APPROX_NONE)
        return contours

    def draw_contours(self, img, contours, fname=None):
        """Draw contours

        @param img: input image array
        @param contours: contours to be drawn

        Keyword arguments:
        fname -- output file name

        """
        for contour in contours:
            [x, y, w, h] = boundingRect(contour)

            if h > len(img)*0.9 or w > len(img[0])*0.9:
                continue
            if h < 4 or w < 4:
                continue
            rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        if fname is not None:
            self.save(img, fname)
        return img

    def satuation(self, img, fname=None):
        """Get the image satuation

        @param img: image array

        Keyword arguments:
        fname -- output file name

        """
        if not self.is_rgb(img):
            print('ERROR: Cannot support grayscale images')
            sys.exit(0)
        np.seterr(divide='ignore')
        sat = 1 - np.divide(3, (img.sum(axis=2)*img.min(axis=2)))
        sat[np.isneginf(sat)] = 0
        if fname is not None:
            self.save(sat, fname)
        return sat

    def intensity(self, img, fname=None):
        """Get the pixel intensity

        @param img: image array

        Keyword arguments:
        fname -- output file name

        """
        if self.is_rgb(img):
            intensity = self.gray(img)
        else:
            intensity = img
        intensity = intensity.astype(float)
        intensity *= (1.0/intensity.max())
        if fname is not None:
            self.save(intensity, fname)
        return intensity

    def is_rgb(self, img):
        """Check if the image is rgb or gray scale"""

        if len(img.shape) > 2 or img.shape[2] == 3:
            return True
        return False

    def gray(self, img, fname=None):
        """Convert the image to gray scale

        @param img: image array

        Keyword arguments:
        fname -- output file name

        """
        gray = cvtColor(img, COLOR_BGR2GRAY)
        if fname is not None:
            self.save(gray, fname)
        return gray

    def save(self, img, fname='cv2.img'):
        """Write images

        @param img: image array

        Keyword arguments:
        fname -- output file name

        """
        imwrite(fname, img)
        return 0

    def read(self, fimg):
        """Access image pixels

        @param fimg: input image file name

        """
        from cv2 import imread
        return imread(fimg)

    def get_jpeg_quality(self, img_path):
        """Get the jpeg quality using identify tool"""

        try:
            q = subprocess.check_output("identify -verbose %s | grep Quality"
                                        % img_path, shell=True)
            q = q.replace(' ', '').split('\n')[0].split(':')[1]
            return int(q)
        except subprocess.CalledProcessError:
            return None
