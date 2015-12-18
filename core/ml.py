import os
import time
import logging
import numpy as np
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
        self._tune_args()

    def _add_args(self):
        """Called by __init__ of Args class"""
        pass

    def _tune_args(self):
        """Called by __init__ of Args class (after _set_args)"""
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
        self.get_prob = True
        self.test_size = 0.33
        self.retrain = True
        self.random = 42
        self.outd = './'

    def tune_args_for_data(self, N):
        """Tunning args right before training is applied

        @param N: number of total data entry

        """
        pass


class NeighborsArgs(MLArgs):
    def _add_args(self):
        """Function to add additional arguments"""

        self._add_neighbors_args()

    def _add_neighbors_args(self):
        """Add additional arguments for SVM class"""

        self._add_ml_args()
        self.algorithm = 'brute'
        self.radius = 0
        self.more = False
        self.n_neighbors = None

    def _tune_args(self):
        """Tune args after running _set_srgs"""

        self.grids = [{'radius': [0.5 * self.radius,
                                  self.radius,
                                  1.5 * self.radius],
                       'weights': ['uniform', 'distance'],
                       'algorithm': [self.algorithm],
                       'leaf_size': [20, 30, 40],
                       'p': [1, 2, 3]}]
        if self.radius > 0:
            self.get_prob = False
        if self.n_neighbors is not None:
            vec = np.arange(0.5 * self.n_neighbors,
                            1.5 * self.n_neighbors,
                            int(self.n_neighbors/4), dtype=int)
            self.grids[0]['n_neighbors'] = vec

    def tune_args_for_data(self, N):
        """Tunning args right before training is applied

        @param N: number of total data entry

        """
        import math
        if self.radius == 0:
            if self.n_neighbors is None:
                if self.more:
                    k = int(N/2)
                else:
                    k = int(math.sqrt(N))
                vec = np.arange(0.5 * k, 1.5 * k, int(k/4), dtype=int)
                self.grids[0]['n_neighbors'] = list(vec)
            del self.grids[0]['radius']

        else:
            if self.radius == -1:
                self.grids[0]['radius'] = int(math.sqrt(N))
            del self.grids[0]['n_neighbors']
            self.grids[0]['outlier_label'] = N


class RFArgs(MLArgs):
    def _add_args(self):
        """Function to add additional arguments"""

        self._add_rf_args()

    def _add_rf_args(self):
        """Add additional arguments for SVM class"""

        self._add_ml_args()
        self.extreme = True
        self.grids = [{'criterion': ['gini', 'entropy'],
                       'bootstrap': [True, False],
                       'random_state': [None, 1, 64],
                       'n_estimators': [64, 96, 128, 256],
                       'max_features': [None, 'sqrt', 'log2']}]


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

    def _tune_args(self):
        """Tune args after running _set_srgs"""

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

    def PCA(self, X, Y=None, ncomp=2, method='PCA'):
        """ decompose a multivariate dataset in an orthogonal
            set that explain a maximum amount of the variance

        @param X: Input dataset

        Keyword Arguments:
        ncomp  -- number or components to be kept (Default: 2)
        method -- method to be used
                  PCA(default)/Randomized/Sparse

        """
        from sklearn import decomposition
        from sklearn import cross_decomposition
        if method == 'Randomized':
            pca = decomposition.RandomizedPCA(n_components=ncomp)
        elif method == 'Sparse':
            pca = decomposition.SparsePCA(n_components=ncomp)
        elif method == 'rbf':
            pca = decomposition.KernelPCA(n_components=ncomp,
                                          fit_inverse_transform=True,
                                          gamma=10, kernel="rbf")
        elif method == 'linear':
            pca = decomposition.KernelPCA(n_components=ncomp,
                                          kernel="linear")
        elif method == 'sigmoid':
            pca = decomposition.KernelPCA(n_components=ncomp,
                                          kernel="sigmoid")
        elif method == 'SVD':
            pca = decomposition.TruncatedSVD(n_components=ncomp)
        else:
            pca = decomposition.PCA(n_components=ncomp)
            method = 'PCA'
        print('Using %s method' % method)
        pca.fit(X)
        return pca.transform(X)


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

    def _set_model(self):
        """Place holder for child class to set the ML model"""

        return None

    def run(self, data, target):
        """Run spliting sample, training and testing

        @param data: Input full data array (multi-dimensional np array)
        @param target: Input full target array (1D np array)

        """
        data = dt.conv_to_np(data)
        target = dt.conv_to_np(target)
        length = dt.check_len(data, target)
        train_d, test_d, train_t, test_t = \
            self.split_samples(data, target)
        model, method = self.train_with_grids(train_d, train_t)
        if len(test_t) > 0 and self.args.retrain:
            result = self.test(test_d, test_t, model)
            if self.args.retrain:
                print("Re-fit model with the full dataset")
                model.fit(data, target)
        else:
            print('No additional testing is performed')
            result = None
        mf = self.save_model(method, model)
        return result

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

    def train_with_grids(self, data, target):
        """Train with GridSearchCV to Find the best parameters

        @param data: Input training data array (multi-dimensional np array)
        @param target: Input training target array (1D np array)

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
        self.args.tune_args_for_data(len(data))
        method, model = self._set_model()
        if model is None:
            print("Error: cannot set the model properly")
            sys.exit(1)
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

        return clf, method

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
        if self.args.get_prob:
            prob = trained_model.predict_proba(data)
        else:
            prob = None
        accuracy = metrics.accuracy_score(target, predicted)
        error = dt.cal_standard_error(predicted)

        print(metrics.classification_report(target, predicted,
                                            target_names=target_names))
        print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy, error))

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

    def _set_model(self):
        """Set ML model"""

        from sklearn import neighbors
        if self.args.radius == 0:
            return 'Neighbors', neighbors.KNeighborsClassifier()

        else:
            return 'Neighbors', neighbors.RadiusNeighborsClassifier()


class RFRun(MLRun):
    def ml_init(self, pfs):
        """Initialize arguments needed

        @param pfs: profiles to be read (used by SVMArgs)

        """
        self.args = RFArgs(pfs=pfs)

    def _set_model(self):
        """Set ML model"""

        from sklearn import ensemble
        if self.args.extreme:
            return 'ExtremeRF', ensemble.ExtraTreesClassifier()
        else:
            return 'RF', ensemble.RandomForestClassifier()


class SVMRun(MLRun):
    def ml_init(self, pfs):
        """Initialize arguments needed

        @param pfs: profiles to be read (used by SVMArgs)

        """
        self.args = SVMArgs(pfs=pfs)

    def _set_model(self):
        """Set ML model"""

        from sklearn import svm
        return 'SVC', svm.SVC(probability=True)
