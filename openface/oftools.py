import os
import sys
import logging
from simdat.core import tools
from simdat.core import plot
from simdat.core import ml

io = tools.MLIO()
pl = plot.PLOT()


class OFArgs(ml.SVMArgs):
    def _add_args(self):
        """Init arguments of openface"""
        self._add_svm_args()
        self._add_of_args()

    def _add_of_args(self):
        """Add additional arguments for OpenFace"""

        self.pathOF = os.path.join(os.getenv("HOME"), 'openface')
        self.pathdlib = os.path.join(os.getenv("HOME"),
                                     'src', 'dlib-18.16',
                                     'python_examples')
        self.parentModel = 'models'
        self.parentdlibModel = 'dlib'
        self.fdlibFaceMean = 'mean.csv'
        self.fdlibPredictor = 'shape_predictor_68_face_landmarks.dat'
        self.parentOFModel = 'openface'
        self.fmodel = 'nn4.v1.t7'
        self.imgDim = 96
        self.cuda = True
        self.outf = './result.json'

        self.pathdlibMean = None
        self.pathPredictor = None
        self.pathModel = None


class OpenFace(ml.SVMRun):
    def ml_init(self, pfs):
        """Init function of OpenFace"""

        self.args = OFArgs(pfs=pfs)
        self.set_paths()

    def set_paths(self):
        """Check paths used by openface"""

        _model_parent = os.path.join(self.args.pathOF,
                                     self.args.parentModel)
        if self.args.pathdlibMean is None:
            self.args.pathdlibMean = os.path.join(self.args.pathOF,
                                                  "./models/dlib/",
                                                  self.args.fdlibFaceMean)
        if self.args.pathPredictor is None:
            self.args.pathPredictor = os.path.join(_model_parent,
                                                   self.args.parentdlibModel,
                                                   self.args.fdlibPredictor)
            self.args.pathPredictor = str(self.args.pathPredictor)
        if self.args.pathModel is None:
            self.args.pathModel = os.path.join(_model_parent,
                                               self.args.parentOFModel,
                                               self.args.fmodel)
        for attr in self.args.__dict__.keys():
            if attr[:4] == 'path':
                io.check_exist(getattr(self.args, attr))
        sys.path.append(self.args.pathdlib)
        return

    def read_df(self, inf, dtype='train'):
        """Read results as Pandas DataFrame

        @param inf: input file to be read

        Keyword arguments:
        dtype -- data type, train or test (default: train)

        """
        df = io.read_json_to_df(inf, orient='index', np=False)
        target = 'class'
        fname = './mapping.json'
        if dtype == 'train':
            mapping = self.map_cats_int(df, groupby=target)
            print("Map of target - int is written to %s" % fname)
            io.write_json(mapping, fname=fname)
        elif dtype == 'test':
            mapping = io.parse_json(fname)
        df[target] = df[target].apply(lambda x: mapping[x])
        data = df['rep'].tolist()
        target = df[target].tolist()
        return data, target

    def map_cats_int(self, df, groupby='class'):
        """Create a mapping for the categories to integers

        @param df: dataframe to map

        Keyword arguments:
        groupby -- keyword of the class to be mapped

        """
        cats = df.groupby(groupby).count().index.tolist()
        return dict(zip(cats, range(0, len(cats))))

    def get_rep(self, imgPath, net=None,
                output=False, class_kwd='person-'):
        """Get facenet representation of a image

        @param imgPath: path of the input image

        Keyword arguments:
        net       -- existing net model
                     (default: None, re-gen from self.get_net())
        output    -- true to output the results (default: False)
                     * This is different from self.get_reps
        class_kwd -- keyword to identify the image class
                     from path (default: person-)

        """
        print("Processing %s" % imgPath)
        io.check_exist(imgPath)
        alignedFace = self.align(imgPath)
        if alignedFace is None:
            return alignedFace
        rep = self.cal_rep(alignedFace, net=net)
        key = io.gen_md5(rep)
        result = {key: {'path': imgPath, 'rep': rep,
                        'dim': self.args.imgDim,
                        'class': self.get_class_from_path(imgPath,
                                                          class_kwd)}}
        if output:
            io.check_parent(self.args.outf)
            io.write_json(result, fname=self.args.outf)
        return result

    def get_reps(self, imgs, net=None,
                 output=True, class_kwd='person-'):
        """Get face representations of multuple images

        @param imgs: a list input image paths

        Keyword arguments:
        net       -- existing net model
                     (default: None, re-gen from self.get_net())
        output    -- true to output the results (default: True)
        class_kwd -- keyword to identify the image class
                     from path (default: person-)
        """
        results = {}
        if net is None:
            net = self.get_net()
        for img in imgs:
            r = self.get_rep(img, net=net, output=output,
                             class_kwd=class_kwd)
            if r is not None:
                results.update(r)
        if output:
            io.check_parent(self.args.outf)
            io.write_json(results, fname=self.args.outf)
        return results

    def align(self, imgPath):
        """Get aligned face of a image

        @param imgPath: input image path

        """
        import dlib
        import cv2
        from openface.alignment import NaiveDlib

        align = NaiveDlib(self.args.pathdlibMean, self.args.pathPredictor)
        img = cv2.imread(imgPath)
        if img is None:
            print("WARNING: Unable to load image: {}".format(imgPath))
            return None

        logging.debug("  + Original size: {}".format(img.shape))
        bb = align.getLargestFaceBoundingBox(img)
        if bb is None:
            print("Unable to find a face: {}".format(imgPath))
            return None

        alignedFace = align.alignImg("affine", self.args.imgDim, img, bb)
        if alignedFace is None:
            print("Unable to align image: {}".format(imgPath))
            return None

        return alignedFace

    def get_net(self):
        """Open the pre-trained net"""

        import openface
        return openface.TorchWrap(self.args.pathModel,
                                  imgDim=self.args.imgDim,
                                  cuda=self.args.cuda)

    def cal_rep(self, alignedFace, net=None):
        """Calculate facenet representation for an aligned face

        @param alignedFace: aligned face

        Keyword arguments:
        net -- pre-opened net

        """
        if net is None:
            net = self.get_net()
        rep = net.forwardImage(alignedFace)
        logging.debug("Representation:")
        logging.debug(rep)
        logging.debug("-----\n")
        return rep
