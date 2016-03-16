import os
import sys
import time
import logging
import numpy as np
from simdat.core import tools
from simdat.core import args

io = tools.MLIO()
dt = tools.DATA()


class DataArgs(args.Args):
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


class MLArgs(args.Args):
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
        self.multiclass = None

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


class MLPArgs(MLArgs):
    def _add_args(self):
        """Function to add additional arguments"""

        self._add_mlp_args()

    def _add_mlp_args(self):
        """Add additional arguments for SVM class"""

        self._add_ml_args()
        self.class_mode = 'categorical'
        self.loss = 'mean_squared_error'
        self.nb_epoch = 100
        self.dropout = 0.5
        self.bsize = 32


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
        """Init function of MLTools class"""

        self.args = MLArgs(pfs=[])
        return

    def get_class_from_path(self, opath, keyword=''):
        """Get object class from the file path

        @param opath: path of the object
        @param keyword: keyword of the classes to search

        """
        _dirname = os.path.dirname(opath)
        while len(_dirname) > 1:
            base = os.path.basename(_dirname)
            if keyword == '':
                return base
            elif keyword is not None and base.find(keyword) > -1:
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
        print('[ML] Using %s method' % method)
        pca.fit(X)
        return pca.transform(X)

    def save_model(self, fprefix, model, high=False):
        """Save model to a file for future use

        @param fprefix: prefix of the output file
        @param model: model to be saved

        """
        import pickle
        io.dir_check(self.args.outd)
        outf = ''.join([self.args.outd, fprefix, '.pkl'])

        with open(outf, 'wb') as f:
            if high:
                pickle.dump(model, f,
                             protocol=pickle.HIGHEST_PROTOCOL)
            else:
                pickle.dump(model, f)
        print("[ML] Model is saved to %s" % outf)
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

    def _init_model(self):
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
        model, method = self.train(train_d, train_t)
        if len(test_t) > 0:
            result = self.test(test_d, test_t, model)
            if self.args.retrain:
                print("[ML] Re-fit model with the full dataset")
                if method == 'MLP':
                    target = dt.convert_cats(target)
                model.fit(data, target)
        else:
            print('[ML] No additional testing is performed')
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

    def train(self, data, target):
        """Train with GridSearchCV to Find the best parameters

        @param data: Input training data array (multi-dimensional np array)
        @param target: Input training target array (1D np array)

        @return clf model trained, path of the output model

        """
        from sklearn import cross_validation
        cv = cross_validation.KFold(len(data),
                                    n_folds=self.args.nfolds)
        self.args.tune_args_for_data(len(data))
        method, model = self._init_model()
        if model is None:
            print("[ML] Error: cannot set the model properly")
            sys.exit(1)

        from sklearn.grid_search import GridSearchCV
        t0 = time.time()
        if 'grids' not in self.args.__dict__.keys():
            raise Exception("grids are not set properly")

        logging.debug('Splitting the jobs into %i' % self.args.njobs)
        log_level = logging.getLogger().getEffectiveLevel()

        def _verbose_level(log_level):
            return int(log_level * (-0.1) + 3)
        verbose = 0 if log_level == 30 else _verbose_level(log_level)

        if model is None:
            print("[ML] Error: cannot set the model properly")
            sys.exit(1)
        print('[ML] GridSearchCV for: %s' % str(self.args.grids))

        clf = GridSearchCV(model, self.args.grids,
                           n_jobs=self.args.njobs,
                           cv=cv, verbose=verbose)
        clf.fit(data, target)
        best_parms = clf.best_params_
        t0 = dt.print_time(t0, 'find best parameters - train')
        print ('[ML] Best parameters are: %s' % str(best_parms))

        best_model = clf.best_estimator_
        if self.args.multiclass is not None:
            clf = self._multiclass_refit(best_model)
            clf.fit(data, target)

        return best_model, method

    def _multiclass_refit(self, clf):
        """Return advanced choices of the classification method"""

        if self.args.multiclass == 'one-vs-rest':
            from sklearn.multiclass import OneVsRestClassifier
            print('[ML] Using one-vs-rest method to re-train')
            clf = OneVsRestClassifier(clf)

        elif self.args.multiclass == 'one-vs-one':
            from sklearn.multiclass import OneVsOneClassifier
            self.args.get_prob = False
            print('[ML] Using one-vs-one method to re-train')
            print('[ML] WARNING: Set get_prob to False')
            clf = OneVsOneClassifier(clf)

        elif self.args.multiclass == 'error-correcting':
            from sklearn.multiclass import OutputCodeClassifier
            print('[ML] Using error-correcting method to re-train')
            clf = OutputCodeClassifier(clf, code_size=2)

        return clf

    def predict(self, data, trained_model, outf=None):
        """Predict using the existing model

        @param data: Input testing data array (multi-dimensional np array)
        @param trained_model: pre-trained model used for predicting

        Keyword arguments:
        outf -- path of the output file (default: no output)

        """
        t0 = time.time()
        result = {'Result': self._get_predicted(data, trained_model)}
        result['predicted'] = result['Result']
        if outf is not None:
            io.write_json(result, fname=outf)
        t0 = dt.print_time(t0, 'predict %i data entries' % len(data))
        return result

    def _get_predicted(self, data, trained_model):
        """Get predicted vector"""
        return trained_model.predict(data)

    def test(self, data, target, trained_model, target_names=None):
        """Test the existing model

        @param data: Input testing data array (multi-dimensional np array)
        @param target: Input testing target array (1D np array)
        @param trained_model: pre-trained model used for testing

        @return a dictionary of accuracy, std error and predicted output

        """
        t0 = time.time()
        from sklearn import metrics
        predicted = self._get_predicted(data, trained_model)
        if self.args.get_prob:
            prob = trained_model.predict_proba(data)
        else:
            prob = None
        accuracy = metrics.accuracy_score(target, predicted)
        error = dt.cal_standard_error(predicted)

        print(metrics.classification_report(target, predicted,
                                            target_names=target_names))
        print("[ML] Accuracy: %0.5f (+/- %0.5f)" % (accuracy, error))

        result = {'accuracy': accuracy, 'error': error,
                  'predicted': predicted, 'prob': prob,
                  'cm': metrics.confusion_matrix(target, predicted)}

        return result


class MLPRun(MLRun):
    def ml_init(self, pfs):
        """Initialize arguments needed

        @param pfs: profiles to be read (used by MLArgs)

        """
        self.args = MLPArgs(pfs=pfs)

    def _init_model(self, parms=None):
        """Create the MLP model"""

        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.optimizers import SGD

        print("[ML] Setting MLP model...")
        model = Sequential()
        model.add(Dense(parms['indim'], init='uniform',
                        input_dim=parms['indim']))
        model.add(Activation('tanh'))
        model.add(Dropout(self.args.dropout))
        model.add(Dense(64, init='uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(self.args.dropout))
        model.add(Dense(parms['ncat'], init='uniform'))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=self.args.loss,
                      optimizer=sgd,
                      class_mode=self.args.class_mode)
        return 'MLP', model

    def train(self, data, target):
        """Train with the Keras MLP model

        @param data: Input training data array (multi-dimensional np array)
        @param target: Input training target array (1D np array)

        @return clf model trained, training method

        """
        data = dt.conv_to_np(data)
        target = dt.convert_cats(target)
        parms = {'ncat': target.shape[1],
                 'indim': data.shape[1]}
        method, model = self._init_model(parms=parms)
        print("[ML] Training MLP")
        model.fit(data, target, nb_epoch=self.args.nb_epoch,
                  batch_size=self.args.bsize,
                  verbose=1, show_accuracy=True)
        return model, method

    def _get_predicted(self, data, trained_model):
        """Get predicted vector"""
        return trained_model.predict_classes(data, verbose=1,
                                             batch_size=self.args.bsize)

    def read_model(self, fmodel, parms):
        """Read model from a file

        @param fmodel: file path of the model weights
        @param parms = {'ncat': target.shape[1],
                        'indim': data.shape[1]}

        """
        if not os.path.isfile(fmodel):
            raise Exception("Model file %s does not exist." % fmodel)

        model = self._init_model(parms=parms)
        model.load_weights(fmodel)
        return model

    def save_model(self, fprefix, model):
        """Save model to a file for future use

        @param fprefix: prefix of the output file
        @param model: model to be saved

        """
        io.dir_check(self.args.outd)
        outf = ''.join([self.args.outd, fprefix, '.pkl'])
        model.save_weights(outf, overwrite=True)
        print("[ML] Model is saved to %s" % outf)
        return outf


class NeighborsRun(MLRun):
    def ml_init(self, pfs):
        """Initialize arguments needed

        @param pfs: profiles to be read (used by NeighborsArgs)

        """
        self.args = NeighborsArgs(pfs=pfs)

    def _init_model(self, parms=None):
        """Set ML model"""

        from sklearn import neighbors
        if self.args.radius == 0:
            if parms is not None:
                model = neighbors.KNeighborsClassifier(**parms)
            else:
                model = neighbors.KNeighborsClassifier()

        else:
            if parms is not None:
                model = neighbors.RadiusNeighborsClassifier(**parms)
            else:
                model = neighbors.RadiusNeighborsClassifier()
        return 'Neighbors', model


class RFRun(MLRun):
    def ml_init(self, pfs):
        """Initialize arguments needed

        @param pfs: profiles to be read (used by RFArgs)

        """
        self.args = RFArgs(pfs=pfs)

    def _init_model(self, parms=None):
        """Set ML model"""

        from sklearn import ensemble
        if self.args.extreme:
            if parms is not None:
                return 'ExtremeRF', ensemble.ExtraTreesClassifier(**parms)
            return 'ExtremeRF', ensemble.ExtraTreesClassifier()
        else:
            if parms is not None:
                return 'RF', ensemble.RandomForestClassifier(**parms)
            return 'RF', ensemble.RandomForestClassifier()


class SVMRun(MLRun):
    def ml_init(self, pfs):
        """Initialize arguments needed

        @param pfs: profiles to be read (used by SVMArgs)

        """
        self.args = SVMArgs(pfs=pfs)

    def _init_model(self, parms=None):
        """Set ML model"""

        from sklearn import svm
        if parms is None:
            parms = {'probability': True}
        else:
            parms['probability'] = True
        return 'SVC', svm.SVC(**parms)
