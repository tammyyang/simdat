import os
import sys
import subprocess
import numpy as np
import logging
import cv2
from PIL import Image
from simdat.core.so import math_tools
from simdat.core import tools
from simdat.core import plot
from simdat.core import args


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

        if dir_path is not None and os.path.isfile(dir_path):
            return [dir_path]
        return self.find_files(dir_path=dir_path, keyword=keyword,
                               suffix=('.jpg', 'png', '.JPEG'))

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

    def find_boundary(self, img, thre=0, findmax=True):
        mean = np.array(img).mean(axis=1)
        selected = [i for i in range(0, len(mean)) if mean[i] > thre]
        start = selected[0]
        end = selected[-1]
        if findmax:
            return max(start, len(img) - end)
        return min(start, len(img) - end)

    def crop_black_bars(self, img, fname=None, thre=1):
        """Crop symmetric black bars"""

        if self.is_rgb:
            _gray = self.gray(img)
        else:
            _gray = img
        cut1 = self.find_boundary(_gray, thre=thre)
        cut2 = self.find_boundary(_gray.T, thre=thre)

        if cut1 > 0:
            img = img[cut1:-cut1]
        if cut2 > 0:
            img = img[:, cut2:-cut2]

        if fname is not None:
            logging.info('Saving croped file as %s.' % fname)
            self.save(img, fname)

        return img

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
                      save=False, rect=False, whratio=-1.0,
                      color=(255, 0, 255), width=2, bcut=0.3, bwidth=0.1):
        """Draw contours

        @param img: input image array
        @param contours: contours to be drawn

        Keyword arguments:
        amin   -- min of the area to be selected
        amax   -- max of the area to be selected
        rect   -- True to draw boundingRec (default: False)
        save   -- True to save the image (default: False)
        color  -- Line color (default: (255, 0, 255))
        width  -- Line width, -1 to fill (default: 2)
        bwidth -- boundary selection width, set it to 0 if no boundary
                  selection should be applied (default: 0.1)
        bcut   -- boundary selection ratio, set it to 0 if no boundary
                  selection should be applied (default: 0.3)

        """
        areas = []
        [h0, w0] = img.shape
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
                if bcut > 0 and bwidth > 0:
                    if w > h:
                        if (y <= h0*bcut and y+h >= h0*(bcut + bwidth)):
                            continue
                        if (y+h >= h0*(1-bcut) and y <= h0*(1-(bcut+bwidth))):
                            continue
                        if (y >= h0*bcut and y+h <= h0*(1-bcut)):
                            continue
                    if h > w:
                        if (w <= w0*bcut and x+w >= w0*(bcut + bwidth)):
                            continue
                        if (x+w >= w0*(1-bcut) and x <= w0*(1-(bcut+bwidth))):
                            continue
                        if (x >= w0*bcut and x+w <= w0*(1-bcut)):
                            continue
                cv2.rectangle(img, (x, y), (x+w, y+h), color, width)
                areas.append([x, y, w, h])

            else:
                cv2.drawContours(img, [cnt], 0, color, width)
                areas.append(area)
        if save:
            self.save(img, 'contours.png')
        return img, areas

    def is_rgb(self, img):
        """Check if the image is rgb or gray scale"""

        if len(img.shape) <= 2:
            return False
        if img.shape[2] < 3:
            return False
        return True

    def substract_bkg(self, img, bkgs, fname=None):
        """Substract image background

        @param img: input forward image in np array
        @param bkgs: a list of background image in np arrays

        Keyword arguments:
        fname -- specify to output the substracted image

        """

        backsub = cv2.BackgroundSubtractorMOG2()
        fgmask = None
        for bkg in bkgs:
            fgmask = backsub.apply(bkg)
        fgmask = backsub.apply(img)
        if fname is not None and type(fname) is str:
            self.save(fgmask, fname=fname)
        return cv2.bitwise_and(img, img, mask=fgmask)

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

    def save(self, img, fname='cv2.jpg'):
        """Write images

        @param img: image array

        Keyword arguments:
        fname -- save file name

        """
        cv2.imwrite(fname, img)
        return 0

    def read_and_random_crop(self, fimg, size=None, ratio=0.7, save=False):
        """Access image pixels

        @param fimg: input image file name

        Keyword arguments:
        size -- tuple of new size (default None)
        ratio -- used to determin the croped size (default 0.7)

        @return imgs: dictionary of the croped images

        """
        img = self.read(fimg)
        if img is None:
            return img
        nrow = len(img)
        ncol = len(img[0])
        imgs = {}
        imgs['crop_img_lt'] = img[0:int(nrow*ratio),
                                  0:int(ncol*ratio)]
        imgs['crop_img_lb'] = img[int(nrow*(1-ratio)):nrow,
                                  0:int(ncol*ratio)]
        imgs['crop_img_rt'] = img[0:int(nrow*ratio),
                                  int(ncol*(1-ratio)):ncol]
        imgs['crop_img_rb'] = img[int(nrow*(1-ratio)):nrow,
                                  int(ncol*(1-ratio)):ncol]
        for corner in imgs:
            if size is not None:
                imgs[corner] = self.resize(imgs[corner], size)
            if save:
                dirname = os.path.dirname(fimg)
                dirname = os.path.join(dirname, 'crops')
                self.check_dir(dirname)

                _fname = os.path.basename(fimg).split('.')
                _fname.insert(-1, '_' + corner + '.')

                fname = ''.join(_fname)
                fname = os.path.join(dirname, fname)
                self.save(imgs[corner], fname=fname)
        return imgs

    def resize(self, img, size):
        """ Resize the image """

        from cv2 import resize
        return resize(img, size)

    def read(self, fimg, size=None):
        """Access image pixels

        @param fimg: input image file name

        Keyword arguments:
        size -- tuple of new size (default None)

        """
        if not self.check_exist(fimg):
            sys.exit(1)

        from cv2 import imread
        img = imread(fimg)
        if img is None:
            print("[IMAGE] Error reading file %s" % fimg)
            return img
        if size is not None:
            img = self.resize(img, size)

        return img

    def get_jpeg_quality(self, img_path):
        """Get the jpeg quality using identify tool"""

        try:
            q = subprocess.check_output("identify -verbose %s | grep Quality"
                                        % img_path, shell=True)
            q = q.replace(' ', '').split('\n')[0].split(':')[1]
            return int(q)
        except subprocess.CalledProcessError:
            return None


class OTDArgs(args.Args):
    def _add_args(self):
        """Called by __init__ of Args class"""
        self.ramin = 0.35
        self.ramax = 0.95
        self.rmor_sel = 0.33
        self.mor_ch = 0.1
        self.mor_cw = 0.1
        self.mor_oh = 0.05
        self.mor_ow = 0.05
        self.mor_ds = 0.04
        self.cwhratio = 1.5
        self.rlbpmin = 0.03
        self.rlbpmax = 0.3


class OverlayTextDetection(IMAGE):
    """This is the implementation of the paper

        Overlay Text Detection in Complex Video Background

    Link of the original paper: http://goo.gl/d3GQ3T

    """
    def img_init(self):
        self.math = math_tools.MathTools()
        self.pl = plot.PLOT()
        self.da = tools.DATA()
        self.args = OTDArgs(pfs=['otd_args.json'])

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

    def cal_side_means(self, img, thre=0.15):
        """Calculate the mean of four sides"""

        upper = int(thre*img.shape[0])
        lower = int((1-thre)*img.shape[0])
        up = np.mean(img[:upper])
        down = np.mean(img[lower:])
        left = np.mean(img[:upper, :thre*img.shape[1]]) + \
               np.mean(img[lower:, :thre*img.shape[1]])
        left /= 2
        right = np.mean(img[:upper, (1-thre)*img.shape[1]:]) + \
                np.mean(img[lower:, (1-thre)*img.shape[1]:])
        right /= 2
        logging.debug("up: %.2f, down: %.2f, left: %.2f, right: %.2f"
                      % (up, down, left, right))

        return (up+down+left+right)/4.0

    def detect_text_area(self, img, save=False):
        """Detect text area"""

        gray = self.gray(img, save=save)
        lmb = self.linked_map_boundary(img, save=save)
        lbp = self.LBP(lmb, subtract=True, save=save)
        # Select only values in the middle range
        lbpmax = np.amax(lbp)
        lbp = self.select(lbp, lbpmax*self.args.rlbpmin,
                          lbpmax*self.args.rlbpmax)
        if save:
            self.pl.plot_matrix(lbp, fname='lbp_cm_selected.png', norm=False,
                                show_text=False, show_axis=False)

        # Apply the Morphological window
        mor = self.morph_dilation(lbp, rs=self.args.mor_ds, save=save)
        mor = self.morph_opening(mor, hr=self.args.mor_oh,
                                 wr=self.args.mor_ow, save=save)
        mor = self.morph_closing(mor, hr=self.args.mor_ch,
                                 wr=self.args.mor_cw, save=save)
        mor_selected = np.where(mor > mor.max()*self.args.rmor_sel, 1, 0)
        if save:
            self.pl.plot_matrix(mor_selected, fname='mor_selected.png',
                                norm=False, show_text=False, show_axis=False)

        # Find max rectangle from mor_selected
        size1, pos1 = self.math.max_size(mor_selected)
        a1 = self.math.area(size1)
        croped1 = img[pos1[0]:pos1[0]+size1[0], pos1[1]:pos1[1]+size1[1]]
        if save:
            self.save(croped1, 'area1.png')

        # find contours
        gray = self.gray(img)
        selected_gray = gray*mor_selected
        selected_gray = selected_gray.astype('uint8')
        contours = self.contours(selected_gray)
        total_area = gray.shape[0]*gray.shape[1]
        # Filter out areas which are too big or too small
        gray, areas = self.draw_contours(gray, contours, amin=-1, amax=-1,
                                         save=save, rect=True,
                                         whratio=self.args.cwhratio)

        # Find max rectangle from contours
        tmp = np.zeros(gray.shape)
        for (x, y, w, h) in areas:
            tmp[y:y+h, x:x+w] = 255
        if save:
            self.save(tmp, 'tmp.png')
        size2, pos2 = self.math.max_size(tmp)
        a2 = self.math.area(size2)
        croped2 = img[pos2[0]:pos2[0]+size2[0], pos2[1]:pos2[1]+size2[1]]
        if save:
            self.save(croped2, 'area2.png')

        # Select the good croped area to output
        amin = total_area*self.args.ramin
        amax = total_area*self.args.ramax
        side_mean = self.cal_side_means(mor_selected)
        total_mean = np.mean(mor_selected)
        # case #1: no counter is found, and a1 is good
        logging.debug('side_mean = %.5f' % side_mean)
        logging.debug('total_mean = %.5f' % total_mean)
        logging.debug('A1/total_area = %.2f' % (float(a1)/total_area))
        logging.debug('A2/total_area = %.2f' % (float(a2)/total_area))
        if a1 < amin and a2 < amin:
            return img
        if a1 > amin and a2 > amax:
            logging.debug('a1 > amin and a2 > amax')
            # If non-zero values are mostly in the center, return a2
            if side_mean <= total_mean:
                return croped2
            return croped1
        # case #2: no counter is found, but a1 is too small
        if a1 <= amin and a2 > amax:
            logging.debug('a1 <= amin and a2 > amax')
            return croped2
        # case #3: a1 is too large but a2 is reasonable
        if a1 > amax and a2 <= amax:
            logging.debug('a1 > amax and a2 <= amax')
            return croped2
        # case #4: a2 is too large but a1 is reasonable
        if a2 > amax and a1 <= amax:
            logging.debug('a2 > amax and a1 <= amax')
            return croped1
        # case #5: a1 and a2 are both reasonable, pick the bigger one
        if a1 > a2:
            logging.debug('a1 > a2')
            return croped1
        return croped2
