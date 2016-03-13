import os
import sys
import logging
from simdat.core import tools
from simdat.core import plot
from simdat.core import ml
from simdat.core import image
from simdat.core import args

io = tools.MLIO()
pl = plot.PLOT()
mlr = ml.MLRun()


class OFArgs(args.Args):
    def _add_args(self):
        """Init arguments of openface"""
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
        self.fmodel = 'nn4.small2.v1.t7'
        self.imgDim = 96
        # tightcrop/affine/perspective/homography
        self.align_method = 'affine'
        self.cuda = True
        self.outf = './result.json'

        self.pathdlibMean = None
        self.dlibFacePredictor = None
        self.networkModel = None


class OFTools(object):
    def __init__(self):
        pass

    def pick_reps(self, dbs, dir_path=None):
        """Pick entries which matched the images existing in
           a specified directory from multiple db json files

        @param dbs: list of the input db

        Keyword arguments:
        dir_path: parent directory of the images

        """
        import pandas as pd
        im = image.IMAGE()
        img_sufs = im.find_images(dir_path=dir_path)
        img_sufs = [im.path_suffix(x) for x in img_sufs]
        df = io.read_jsons_to_df(dbs, orient='index')
        df['path_suf'] = df['path'].apply(lambda x: im.path_suffix(x))
        df = df[df['path_suf'].isin(img_sufs)]
        io.write_df_json(df, fname='./picked_rep.json')
        return df

    def read_df(self, inf, dtype='test', mpf='./mapping.json',
                group=False, selclass=None, conv=True):
        """Read results as Pandas DataFrame

        @param inf: input file path to be read or a read df

        Keyword arguments:
        dtype -- data type, train or test (default: test)
        mpf   -- file name of the mapping (default: ./mapping.json)
        group -- true to group data by path (default: False)
        conv  -- true to convert class to int according to mpf

        @return results after reading the input db file
                if group:
                    result['data'] = [list[data_from_image_1],
                                      list[data_from_image_2],.. etc]
                else:
                    result['data'] = [data_from_image_1,
                                      data_from_image_2, .. etc]
                same for 'pos' and 'target'
                result['path'] = list[paths of images]
                result['target_names'] = keys of the mapping file

        """
        if type(inf) is str:
            df = io.read_json_to_df(inf, orient='index', np=False)
        else:
            df = inf
        _target = 'class'

        if dtype == 'train':
            mapping = self.map_cats_int(df, groupby=_target)
            print("Map of target - int is written to %s" % mpf)
            io.write_json(mapping, fname=mpf)
        elif dtype == 'test':
            mapping = io.parse_json(mpf)

        if conv:
            df[_target] = df[_target].apply(lambda x: mapping[x])
        if conv and type(selclass) is int:
            df = df[df['class'] == selclass]
        res = {'data': [], 'target': [], 'pos': [],
               'path': [], 'mapping': mapping}
        if group:
            grouped = df.groupby('path')
            for name, group in grouped:
                gdf = grouped.get_group(name)
                res['data'].append(gdf['rep'].tolist())
                res['pos'].append(gdf['pos'].tolist())
                res['target'].append(gdf[_target].tolist())
                res['path'].append(gdf['path'][0])
        else:
            res['data'] = df['rep'].tolist()
            res['pos'] = df['pos'].tolist()
            res['target'] = df[_target].tolist()
            res['path'] = df['path'].tolist()
        res['target_names'] = self.mapping_keys(mapping)
        return res

    def mapping_keys(self, mapping):
        """Sort mapping dictionaty and get the keys"""

        from operator import itemgetter
        _labels = sorted(mapping.items(), key=itemgetter(1))
        return [i[0] for i in _labels]

    def map_cats_int(self, df, groupby='class'):
        """Create a mapping for the categories to integers

        @param df: dataframe to map

        Keyword arguments:
        groupby -- keyword of the class to be mapped

        """
        cats = df.groupby(groupby).count().index.tolist()
        return dict(zip(cats, range(0, len(cats))))


class OpenFace(OFTools):
    def __init__(self, pfs, method='SVC'):
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
        if self.args.dlibFacePredictor is None:
            self.args.dlibFacePredictor = os.path.join(_model_parent,
                                                       self.args.parentdlibModel,
                                                       self.args.fdlibPredictor)
            self.args.dlibFacePredictor = str(self.args.dlibFacePredictor)
        if self.args.networkModel is None:
            self.args.networkModel = os.path.join(_model_parent,
                                                  self.args.parentOFModel,
                                                  self.args.fmodel)
        for attr in self.args.__dict__.keys():
            if attr[:4] == 'path':
                io.check_exist(getattr(self.args, attr))
        sys.path.append(self.args.pathdlib)
        return

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
        alignedFaces = self.align(imgPath)
        if alignedFaces is None:
            return alignedFaces
        result = {}
        for face in alignedFaces:
            rep = self.cal_rep(face[0], net=net)
            key = io.gen_md5(rep)
            result[key] = {'path': imgPath, 'rep': rep,
                           'dim': self.args.imgDim,
                           'pos': face[1],
                           'class': mlr.get_class_from_path(imgPath,
                                                            class_kwd)}
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
        """Get aligned face(s) of a image

        @param imgPath: input image path

        """
        import dlib
        import cv2
        import openface
        from openface import AlignDlib

        align = AlignDlib(self.args.dlibFacePredictor)
        img = cv2.imread(imgPath)
        if img is None:
            print("Fail to read image: {}".format(imgPath))
            return None

        logging.debug("  + Original size: {}".format(img.shape))
        bbs = align.getAllFaceBoundingBoxes(img)
        if bbs is None:
            print("Fail to detect faces in image: {}".format(imgPath))
            return None

        alignedFaces = []
        logging.debug("Align the face using %s method"
                      % self.args.align_method)
        for bb in bbs:
            alignedFace = align.align(self.args.imgDim, img, bb,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                continue
            alignedFaces.append([alignedFace, [bb.left(), bb.top(),
                                               bb.right(), bb.bottom()]])

            if len(alignedFace) < 1:
                print("Fail to align image: {}".format(imgPath))
                return None

        return alignedFaces

    def get_net(self):
        """Open the pre-trained net"""

        import openface
        return openface.TorchNeuralNet(self.args.networkModel,
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
        rep = net.forward(alignedFace)
        logging.debug("Representation:")
        logging.debug(rep)
        logging.debug("-----\n")
        return rep
