import os
import subprocess
import numpy as np
import logging
import cv2
from PIL import Image
from simdat.core import tools
from simdat.core import plot


class IMAGE(tools.TOOLS):
    def tools_init(self):
        self.img_init()

    def img_init(self):
        pass

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

        if self.is_rgb:
            _gray = self.gray(img)
        else:
            _gray = _gray
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

        return cut

    def laplacian(self, img, save=False):
        """Laplacian transformation"""

        if self.is_rgb(img):
            img = self.gray(img)
        la = cv2.Laplacian(img, cv2.CV_64F)
        if save:
            self.save(la, 'laplacian.png')
        return la

    def sobel(self, img, axis=0, save=False):
        """Sobel transformation"""

        if axis == 0:
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        elif axis == 1:
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        if save:
            self.save(sobel, 'sobel.png')
        return sobel

    def contours(self, img, save=False):
        """Get contours"""

        if self.is_rgb(img):
            img = self.gray(img)
        contours, hier = cv2.findContours(img, cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_contours(self, img, contours, amin=-1, amax=-1,
                      save=False, rect=False, whratio=-1.0):
        """Draw contours

        @param img: input image array
        @param contours: contours to be drawn

        Keyword arguments:
        amin   -- min of the area to be selected
        amax   -- max of the area to be selected
        rect   -- True to draw boundingRec (default: False)
        save -- True to save the image (default: False)

        """
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if amin > 0 and area < amin:
                continue
            if amax > 0 and area > amax:
                continue
            if rect:
                [x, y, w, h] = cv2.boundingRect(cnt)
                if whratio > 0:
                    if w/h < whratio and h/w < whratio:
                        continue
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
            else:
                cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
        if save:
            self.save(img, 'contours.png')
        return img

    def is_rgb(self, img):
        """Check if the image is rgb or gray scale"""

        if len(img.shape) <= 2:
            return False
        if img.shape[2] < 3:
            return False
        return True

    def get_houghlines(self, img):
        """Get lines from hough transform"""

        if self.is_rgb(img):
            img = self.gray(img)

        edges = cv2.Canny(img, 100, 200)
        return cv2.HoughLines(edges, 1, np.pi/180, 200)

    def draw_houghlines(self, img, lines, save=False):
        """Draw lines found by hough transform"""

        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if save:
            self.save(img, 'houghlines.png')
        return lines

    def check_cnt_std(self, img, cnt, thre=0.01):
        """Check if std of contour points is within a threshold"""

        std_x = np.std(cnt.T[0][0])
        w = img.shape[1]
        std_y = np.std(cnt.T[1][0])
        h = img.shape[0]
        if std_x <= thre*w or std_y <= thre*h:
            return False
        return True

    def gray(self, img, save=False):
        """Convert the image to gray scale

        @param img: image array

        Keyword arguments:
        save -- True to save the image

        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if save:
            self.save(gray, 'gray.png')
        return gray

    def select(self, img, imin, imax, default=0, inv=False):
        """Select only values in a given range
           and apply default value to the rest

        @param img: image array
        @param imin: lower limit
        @param imax: upper limit

        Keyword arguments:
        default -- the default value to be applied (default: 0)
        inv     -- invert the selection, to select values NOT
                   in the region (default: False)

        """
        if inv:
            cp_img = np.where(img < imax and img > imin, default, img)
        else:
            cp_img = np.where(img > imax, default, img)
            cp_img = np.where(cp_img < imin, default, cp_img)
        return cp_img

    def LBP(self, img, save=False, parms=None, subtract=False):
        """Get the LBP image
        (reference: http://goo.gl/aeADZd)

        @param img: image array

        Keyword arguments:
        save    -- True to save the image
        parms     -- [points, radius] (default: None)
        subtract -- True to subtract values to pts (default: False)

        """
        from skimage.feature import local_binary_pattern
        if self.is_rgb(img):
            img = self.gray(img)
        if parms is None:
            pts = int(img.shape[0]*img.shape[1]*0.0003)
            radius = min(img.shape[0], img.shape[1])*0.015
        else:
            pts = parms[0]
            radius = parms[1]
        lbp = local_binary_pattern(img, pts, radius,  method='uniform')
        if subtract:
            lbp = np.abs(lbp - pts)
        if save:
            self.pl.plot_matrix(lbp, fname='lbp_cm.png', show_text=False,
                                show_axis=False, norm=False)
        return lbp

    def morph_opening(self, img, hr=0.05, wr=0.1, save=False):
        """Apply Morphological opening transform

        @param img: image array

        Keyword arguments:
        hr   -- ratio to the height, for closing window (default: 0.1)
        wr   -- ratio to the width, for closing window (default: 0.2)
        save -- True to save the image

        """
        h = int(img.shape[0]*hr)
        w = int(img.shape[1]*wr)
        kernel = np.ones((h, w), np.uint8)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        if save:
            self.pl.plot_matrix(opening, fname='opening_cm.png',
                                show_text=False,
                                show_axis=False, norm=False)
        return opening

    def morph_dilation(self, img, rs=0.01, save=False):
        """Apply Morphological dilation transform

        @param img: image array

        Keyword arguments:
        shape -- width of the kernel
        save -- True to save the image

        """
        shape = int(min(img.shape[0], img.shape[1])*rs)
        kernel = np.ones((shape, shape), np.uint8)
        dil = cv2.dilate(img, kernel, iterations=1)
        if save:
            self.pl.plot_matrix(dil, fname='dil_cm.png',
                                show_text=False,
                                show_axis=False, norm=False)
        return dil

    def morph_closing(self, img, hr=0.1, wr=0.2, save=False):
        """Apply Morphological closing transform

        @param img: image array

        Keyword arguments:
        hr   -- ratio to the height, for closing window (default: 0.1)
        wr   -- ratio to the width, for closing window (default: 0.2)
        save -- True to save the image

        """
        h = int(img.shape[0]*hr)
        w = int(img.shape[1]*wr)
        kernel = np.ones((h, w), np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        if save:
            self.pl.plot_matrix(closing, fname='closing_cm.png',
                                show_text=False,
                                show_axis=False, norm=False)
        return closing

    def intensity(self, img, save=False):
        """Get the pixel intensity

        @param img: image array

        Keyword arguments:
        save -- True to save the image

        """
        if self.is_rgb(img):
            intensity = self.gray(img)
        else:
            intensity = img
        intensity = intensity.astype(float)
        intensity *= (1.0/intensity.max())
        if save:
            self.pl.plot_matrix(intensity, fname='intensity_cm.png',
                                show_text=False, show_axis=False, norm=False)
        return intensity

    def save(self, img, fname='cv2.img'):
        """Write images

        @param img: image array

        Keyword arguments:
        fname -- save file name

        """
        cv2.imwrite(fname, img)
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


class OverlayTextDetection(IMAGE):
    """This is the implementation of the paper

        Overlay Text Detection in Complex Video Background

    Link of the original paper: http://goo.gl/d3GQ3T

    """
    def img_init(self):
        self.pl = plot.PLOT()

    def satuation(self, img, save=False):
        """Get the image satuation

        @param img: image array

        Keyword arguments:
        save  -- True to save the image (default: False)

        """
        if not self.is_rgb(img):
            print('ERROR: Cannot support grayscale images')
            sys.exit(0)
        np.seterr(divide='ignore')
        sat = 1 - np.divide(3, (img.sum(axis=2)*img.min(axis=2)))
        sat[np.isneginf(sat)] = 0
        if save:
            self.pl.plot_matrix(sat, fname='sat_cm.png',
                                show_text=False, show_axis=False, norm=False)
        return sat

    def maxS(self, img, save=False):
        """Get maxS, more details see http://goo.gl/d3GQ3T

        @param img: image array

        Keyword arguments:
        save     -- True to save the image (default: False)

        """
        intensity = self.intensity(img, save=save)
        maxS = np.where(intensity > 0.5, 2*(0.5-intensity), 2*intensity)
        if save:
            self.pl.plot_matrix(maxS, fname='maxS_cm.png',
                                show_text=False, show_axis=False, norm=False)
        return maxS

    def tildeS(self, img, save=False, nan_to_num=True):
        """Get tilde S, more details see http://goo.gl/d3GQ3T

        @param img: image array

        Keyword arguments:
        save     -- True to save the image (default: False)
        nan_to_num -- True to convert inf to numbers (default: True)

        """
        sat = self.satuation(img, save=save)
        maxS = self.maxS(img, save=save)
        tildeS = sat/maxS
        if nan_to_num:
            tildeS = np.nan_to_num(tildeS)
        if save:
            self.save(tildeS, 'tildeS_cm.png')
        return tildeS

    def calD(self, diff_tildeS, diff_int, left=True):
        """Get D, more details see http://goo.gl/d3GQ3T

        @param diff_tildeS: difference of tilde S matrix
        @param diff_int: difference of the intensity matrix

        Keyword arguments:
        left  -- True to make D_L, False for D_R

        """
        if left:
            tildeS = np.insert(diff_tildeS, 0,
                               diff_tildeS.T[0], axis=1)
            intensity = np.insert(diff_int, 0,
                                  diff_int.T[0], axis=1)

        else:
            tildeS = np.insert(diff_tildeS, diff_tildeS.shape[1],
                               diff_tildeS.T[-1], axis=1)
            intensity = np.insert(diff_int, diff_int.shape[1],
                                  diff_int.T[-1], axis=1)
        return (1 + tildeS)*intensity

    def T(self, img, save=False, T_H=1):
        """Get D, more details see http://goo.gl/d3GQ3T

        @param img: image array

        Keyword arguments:
        T_H   -- threshold used for the transition map
        save -- True to save the image

        """
        tildeS = self.tildeS(img, save=save)
        intensity = self.intensity(img, save=save)
        diff_tildeS = np.diff(tildeS)
        diff_int = np.absolute(np.diff(intensity))
        D_L = self.calD(diff_tildeS, diff_int) + 1
        D_R = self.calD(diff_tildeS, diff_int, left=False)
        T = np.where(D_R > D_L, 1, 0)
        if save:
            self.pl.plot_matrix(T, fname='T_cm.png', show_text=False,
                                show_axis=False, norm=False)
        return T

    def linked_map_boundary(self, img, save=False,
                            T_H=1, r=0.04):
        """Get linked_map_boundary

        @param img: image array

        Keyword arguments:
        r      -- ratio for setting threshold (default: 0.04)
        T_H    -- threshold used for the transition map
                  (used by self.T)
        save -- True to save the image

        """
        T = self.T(img, save=save, T_H=T_H)
        thre = int(T.shape[1]*r)
        for rth in range(0, T.shape[0]):
            non_zero = np.nonzero(T[rth])[0]
            for i in range(0, len(non_zero) - 1):
                s = non_zero[i]
                e = non_zero[i+1]
                if e - s < thre:
                    T[rth][s:e+1] = 255
        if save:
            self.save(T, 'lmb.png')
        return T

    def detect_text_area(self, img, save=False):
        """Detect text area"""

        lmb = self.linked_map_boundary(img, save=save)
        lbp = self.LBP(lmb, subtract=True, save=save)
        # Select only values in the middle range
        thre = np.amax(lbp)*0.3
        lbp = self.select(lbp, thre*0.1, thre)
        if save:
            self.pl.plot_matrix(lbp, fname='lbp_cm_selected.png', norm=False,
                                show_text=False, show_axis=False)

        # Apply the Morphological window
        mor = self.morph_dilation(lbp, rs=0.04, save=save)
        mor = self.morph_opening(mor, hr=0.05, wr=0.05, save=save)
        mor = self.morph_closing(mor, hr=0.1, wr=0.1, save=save)
        mor_selected = np.where(mor > mor.max()*0.33, 1, 0)
        mor_selected = np.repeat(mor_selected, 3).reshape(img.shape[0],
                                                          img.shape[1], 3)
        final = mor_selected*img
        self.save(final, 'final.png')

        # Draw contours
        final = final.astype('uint8')
        contours = self.contours(self.gray(final))
        total_area = img.shape[0]*img.shape[1]
        # Filter out areas which are too big or too small
        amin = total_area*0.005
        amax = total_area*0.5
        # Draw contours with the cut to w/h and h/w
        self.draw_contours(img, contours, amin=amin, amax=amax,
                           save=save, rect=True, whratio=1.5)

        return img
