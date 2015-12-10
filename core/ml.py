import os
import time
import math
import logging
from simdat.core import tools

io = tools.MLIO()
dt = tools.DATA()


class Args(object):
    def __init__(self, pfs=['ml.json']):
        """Init function of Args

        Keyword arguments:
        pfs -- profiles to read (default: ['ml.json'])

        """
        self._add_args()
        for f in pfs:
            self._set_args(f)

    def _add_args(self):
        """Called by __init__ of Args class"""
        pass

    def _set_args(self, f):
        """Read parameters from profile

        @param f: profile file

        """

        if not os.path.isfile(f):
            print("WARNING: File %s does not exist" % f)
            return
        inparm = io.parse_json(f)
        cinst = self.__dict__.keys()
        for k in inparm:
            if k in cinst:
                setattr(self, k, inparm[k])


class DataArgs(Args):
    def _add_args(self):
        """Called by __init__ of Args class"""

        self.target = None
        self.target_bin = True
        self.inc_flat = True
        self.label = 'trend'
        self.flat_thre = 0.002
        self.shift = 1
        self.norm = 'l2'
        self.extend_by_shift = True

        self._add_da_args()

    def _add_da_args(self):
        """Add additional arguments"""
        pass


class MLArgs(Args):
    def _add_args(self):
        """Called by __init__ of Args class"""

        self._add_ml_args()

    def _add_ml_args(self):
        """Add additional arguments for MLRun class"""

        self.njobs = 4
        self.nfolds = 5
        self.test_size = 0.33
        self.random = 42
        self.outd = './'


class NeighborsArgs(MLArgs):
    def _add_args(self):
        """Function to add additional arguments"""

        self._add_neighbors_args()

    def _add_neighbors_args(self):
        """Add additional arguments for SVM class"""

        self._add_ml_args()
        self.grids = [{'n_neighbors': 0,
                       'weights': ['uniform', 'distance'],
                       'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                       'leaf_size': [20, 30, 40],
                       'p': [1, 2, 3, 4]}]
        self.more = False


class RFArgs(MLArgs):
    def _add_args(self):
        """Function to add additional arguments"""

        self._add_rf_args()

    def _add_rf_args(self):
        """Add additional arguments for SVM class"""

        self._add_ml_args()
        self.grids = [{'max_depth': [3, 4, 5, 6, 7],
                       'n_estimators': [5, 10, 15],
                       'max_features': [1, 2, 'sqrt', 'log2']}]


class SVMArgs(MLArgs):
    def _add_args(self):
        """Function to add additional arguments"""

        self._add_svm_args()

    def _add_svm_args(self):
        """Add additional arguments for SVM class"""

        self._add_ml_args()
        self.kernel = 'rbf'
        self.degree = 3
        self.C = [0.1, 1, 10, 100, 1000]
        self.grids = self._set_grids()
        self._set_grids_C()

    def _set_grids(self):
        """Set SVM grids for GridSearchCV

        @return a dictionary of grid parameters

        """

        grids = [{'kernel': ['rbf'],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]},
                 {'kernel': ['linear']},
                 {'kernel': ['poly'],
                  'coef0': [1, 10, 100, 1000],
                  'degree': [1, 2, 3, 4]},
                 {'kernel': ['sigmoid'],
                  'coef0': [1, 10, 100, 1000]}]
        if self.kernel != 'auto':
            for g in grids:
                if g['kernel'][0] == self.kernel:
                    return [g]
        return grids

    def _set_grids_C(self):
        """Set C for grids"""

        from copy import deepcopy
        if type(self.C) is not list:
            self.C = [self.C]
        for i in range(0, len(self.grids)):
            if 'C' not in self.grids[i]:
                self.grids[i]['C'] = deepcopy(self.C)


class MLTools():
    def __init__(self):
        """Init if MLTools, don't do anything here"""

        return

    def get_class_from_path(self, opath, keyword):
        """Get object class from the file path

        @param opath: path of the object
        @param keyword: keyword of the classes to search

        """
        _dirname = os.path.dirname(opath)
        while len(_dirname) > 1:
            base = os.path.basename(_dirname)
            if base.find(keyword) > -1:
                return base
            _dirname = os.path.dirname(_dirname)
        return None


class MLRun(MLTools):
    def __init__(self, pfs=['ml.json']):
        """Init function of MLRun class

        Keyword arguments:
        pfs -- profiles to read (default: ['ml.json'])

        """
        self.ml_init(pfs)

    def ml_init(self, pfs):
        """Initialize arguments needed

        @param pfs: profiles to be read (used by MLArgs)

        """
        self.args = MLArgs(pfs=pfs)

    def _run(self, data, target, method='SVC'):
        """Run spliting sample, training and testing

        @param data: Input full data array (multi-dimensional np array)
        @param target: Input full target array (1D np array)

        Keyword arguments:
        method -- machine learning method to be used (default: svm.SVC)

        """
        data = dt.conv_to_np(data)
        target = dt.conv_to_np(target)
        length = dt.check_len(data, target)
        train_d, test_d, train_t, test_t = \
            self.split_samples(data, target)
        model, mf = self.train_with_grids(train_d, train_t, method)
        if len(test_t) > 0:
            result = self.test(test_d, test_t, model)
        else:
            print('No additional testing is performed')
        return mf

    def split_samples(self, data, target):
        """Split samples

        @param data: Input full data array (multi-dimensional np array)
        @param target: Input full target array (1D np array)

        """
        from sklearn import cross_validation
        train_d, test_d, train_t, test_t = \
            cross_validation.train_test_split(data, target,
                                              test_size=self.args.test_size,
                                              random_state=self.args.random)
        return train_d, test_d, train_t, test_t

    def train_with_grids(self, data, target, method):
        """Train with GridSearchCV to Find the best parameters

        @param data: Input training data array (multi-dimensional np array)
        @param target: Input training target array (1D np array)
        @param method: machine learning method to be used

        @return clf model trained, path of the output model

        """

        t0 = time.time()
        if 'grids' not in self.args.__dict__.keys():
            raise Exception("grids are not set properly")

        from sklearn import cross_validation
        from sklearn.grid_search import GridSearchCV

        print_len = 50
        logging.debug('Splitting the jobs into %i' % self.args.njobs)
        log_level = logging.getLogger().getEffectiveLevel()

        def _verbose_level(log_level):
            return int(log_level * (-0.1) + 3)
        verbose = 0 if log_level == 30 else _verbose_level(log_level)

        cv = cross_validation.KFold(len(data),
                                    n_folds=self.args.nfolds)
        if method == 'SVC':
            from sklearn import svm
            model = svm.SVC(probability=True)
        elif method == 'RF':
            from sklearn import ensemble
            model = ensemble.RandomForestClassifier()
        elif method == 'Neighbors':
            import numpy as np
            from sklearn import neighbors
            if self.args.grids[0]['n_neighbors'] == 0:
                if self.args.more:
                    k = int(len(data)/2)
                else:
                    k = int(math.sqrt(len(data)))
                vec = np.arange(k - k/2, k + k/2, int(k/4), dtype=int)
                self.args.grids[0]['n_neighbors'] = list(vec)
                model = neighbors.KNeighborsClassifier()
        else:
            from sklearn import svm
            model = sklearn.svm.SVC(probability=True)
        print('GridSearchCV for: %s' % str(self.args.grids))
        clf = GridSearchCV(model, self.args.grids,
                           n_jobs=self.args.njobs,
                           cv=cv, verbose=verbose)

        logging.debug('First %i samples of training data' % print_len)
        logging.debug(str(data[:print_len]))
        logging.debug('First %i samples of training target' % print_len)
        logging.debug(str(target[:print_len]))

        clf.fit(data, target)
        best_parms = clf.best_params_
        t0 = dt.print_time(t0, 'find best parameters - train_with_grids')
        print ('Best parameters are: %s' % str(best_parms))
        mf = self.save_model(method, clf)

        return clf, mf

    def save_model(self, fprefix, model):
        """Save model to a file for future use

        @param fprefix: prefix of the output file
        @param model: model to be saved

        """
        import pickle
        io.dir_check(self.args.outd)
        outf = ''.join([self.args.outd, fprefix, '.pkl'])

        with open(outf, 'wb') as f:
            pickle.dump(model, f)
        print("Model is saved to %s" % outf)
        return outf

    def read_model(self, fmodel):
        """Read model from a file

        @param fmodel: file path of the input model

        """
        if not os.path.isfile(fmodel):
            raise Exception("Model file %s does not exist." % fmodel)

        import pickle
        with open(fmodel, 'rb') as f:
            model = pickle.load(f)
        return model

    def predict(self, data, trained_model, outf=None):
        """Predict using the existing model

        @param data: Input testing data array (multi-dimensional np array)
        @param trained_model: pre-trained model used for predicting

        Keyword arguments:
        outf -- path of the output file (default: no output)

        """
        t0 = time.time()
        result = {'Result': trained_model.predict(data)}
        if outf is not None:
            io.write_json(result, fname=outf)
        t0 = dt.print_time(t0, 'predict %i data entries' % len(data))
        return result

    def test(self, data, target, trained_model, target_names=None):
        """Test the existing model

        @param data: Input testing data array (multi-dimensional np array)
        @param target: Input testing target array (1D np array)
        @param trained_model: pre-trained model used for testing

        @return a dictionary of accuracy, std error and predicted output

        """
        t0 = time.time()
        from sklearn import metrics
        print_len = 50
        predicted = trained_model.predict(data)
        prob = trained_model.predict_proba(data)
        accuracy = metrics.accuracy_score(target, predicted)
        error = dt.cal_standard_error(predicted)

        logging.debug(metrics.classification_report(target, predicted,
                                                    target_names=target_names))
        logging.debug("Accuracy: %0.5f (+/- %0.5f)" % (accuracy, error))

        result = {'accuracy': accuracy, 'error': error,
                  'predicted': predicted, 'prob': prob,
                  'cm': metrics.confusion_matrix(target, predicted)}

        logging.debug('First %i results from the predicted' % print_len)
        logging.debug(str(predicted[:print_len]))
        logging.debug('First %i results from the testing target' % print_len)
        logging.debug(str(target[:print_len]))

        return result


class NeighborsRun(MLRun):
    def ml_init(self, pfs):
        """Initialize arguments needed

        @param pfs: profiles to be read (used by SVMArgs)

        """
        self.args = NeighborsArgs(pfs=pfs)

    def run(self, data, target):
        """Run spliting sample, training and testing

        @param data: Input full data array (multi-dimensional np array)
        @param target: Input full target array (1D np array)

        """

        return self._run(data, target, method='Neighbors')


class RFRun(MLRun):
    def ml_init(self, pfs):
        """Initialize arguments needed

        @param pfs: profiles to be read (used by SVMArgs)

        """
        self.args = RFArgs(pfs=pfs)

    def run(self, data, target):
        """Run spliting sample, training and testing

        @param data: Input full data array (multi-dimensional np array)
        @param target: Input full target array (1D np array)

        """

        return self._run(data, target, method='RF')


class SVMRun(MLRun):
    def ml_init(self, pfs):
        """Initialize arguments needed

        @param pfs: profiles to be read (used by SVMArgs)

        """
        self.args = SVMArgs(pfs=pfs)

    def run(self, data, target):
        """Run spliting sample, training and testing

        @param data: Input full data array (multi-dimensional np array)
        @param target: Input full target array (1D np array)

        """

        return self._run(data, target, method='SVC')
