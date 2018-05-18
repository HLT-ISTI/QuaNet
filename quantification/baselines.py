from os.path import join, exists
import subprocess
from subprocess import PIPE, STDOUT
import numpy as np
import tempfile
from sklearn.datasets import dump_svmlight_file
from data.dataset_loader import parse_dataset_code
import sys
from quantification.constants import SVMPERF_BASE
from quantification.helpers import split_pos_neg, sample_data, define_prev_range, mse, mae, mrae, mnkld, loadDataset
from random import randint
import itertools
from sklearn.externals.joblib import Parallel, delayed
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from abc import ABCMeta, abstractmethod


class BaseQuantifier(metaclass=ABCMeta):
    """
    Abstract quantifier operating with document-by-term matrices
    """
    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X, y=None, *args):...

    def print(self, msg, end='\n'):
        if self.verbose:
            print(msg,end=end)

    def set_params(self, **parameters):
        self.learner.set_params(**parameters)

    def optim(self, Xtr, ytr, Xva, yva, error,
              n_samples,
              sample_size,
              prev_range,
              parameters,
              n_jobs=-1,
              refit=True):

        #create n_samples of sample_length
        Xva_pos, Xva_neg,_,_ = split_pos_neg(Xva,yva)
        samples_prevalences = np.repeat(prev_range, n_samples / prev_range.size)
        samples = [sample_data(Xva_pos, Xva_neg, p, sample_size) for p in samples_prevalences] # each sample is (X,y,real_p)
        true_prevs = [p for _,_,p in samples]

        verb = self.verbose
        self.verbose = False

        params_keys = list(parameters.keys())
        params_values = list(parameters.values())

        best_p, best_error = None, None
        for values_ in itertools.product(*params_values):
            params_ = {k:values_[i] for i,k in enumerate(params_keys)}
            print('checking params={} ... '.format(params_), end='')
            sys.stdout.flush()

            self.set_params(**{**self.default_parameters, **params_})
            self.fit(Xtr,ytr)
            estim_prevs = Parallel(n_jobs=n_jobs)(delayed(self.predict)(X,y) for X,y,p in samples)
            score = error(true_prevs, estim_prevs)

            print('\tgot {} score {:.5f}'.format(error.__name__, score), end='')
            if best_error is None or score < best_error:
                best_error, best_p = score, params_
                print('\t[best found]', end='')
            print()

        self.set_params(**{**self.default_parameters, **best_p})
        self.verbose = verb

        if refit:
            print('refitting for {}...'.format(best_p))
            self.fit(Xtr,ytr)


class EM_Quantifier(BaseQuantifier):
    def __init__(self, probabilistic_learner, **kwargs):
        assert hasattr(probabilistic_learner, 'predict_proba'), \
            'learner {} is not probabilistic'.format(probabilistic_learner.__name__)
        self.default_parameters = kwargs
        self.learner = probabilistic_learner(kwargs)
        self.verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        self.__name__ = 'EQ-quantifier'

    def fit(self, X, y):
        self.learner.fit(X,y)
        self.ytr=y
        self.Xtr=X

    def predict(self, X, y=None, epsilon=1e-3):
        def prevalence(y, classname=1):
            return (y == classname).sum() / y.size

        #Pxtr = self.learner.predict_proba(self.Xtr)
        Pxtr = self.learner.predict_proba(X)
        Pxtr_pos = Pxtr[:, self.learner.classes_ == 1].flatten()
        Pxtr_neg = Pxtr[:, self.learner.classes_ != 1].flatten()
        ytr = self.ytr
        trueprev = prevalence(y) if y is not None else -1

        Ptr_pos = prevalence(ytr, classname=1) #Ptr(y=+1)
        Ptr_neg = prevalence(ytr, classname=0) #Ptr(y=0)
        qs_pos,qs_neg = Ptr_pos, Ptr_neg   # i.e., prevalence(ytr)

        s,converged = 0,False
        qs_pos_prev_ = None
        while not converged and s < 20:
            # E-step: ps is Ps(y=+1|xi)
            pos_factor = (qs_pos / Ptr_pos) * Pxtr_pos
            neg_factor = (qs_neg / Ptr_neg) * Pxtr_neg
            ps = pos_factor / (pos_factor + neg_factor)

            # M-step: qs_pos is Ps+1(y=+1)
            qs_pos = np.mean(ps)

            self.print(('s={} qs_pos={:.6f}'+('' if y is None else ' true={:.6f}'.format(trueprev))).format(s,qs_pos))

            if qs_pos_prev_ is not None and abs(qs_pos - qs_pos_prev_) < epsilon and s>10:
                converged = True

            qs_pos_prev_ = qs_pos
            s += 1
        self.print('-'*80)

        return qs_pos


class SVMperfQuantifier(BaseQuantifier):

    # losses with their respective codes in svm_perf implementation
    valid_losses = {'01':0,'kld':12,'nkld':13,'q':22,'qacc':23,'qf1':24,'qgm':25} #13,22

    def __init__(self, svmperf_base, C=0.01, verbose=True, loss='01'):
        assert loss in self.valid_losses, 'unsupported loss {}, valid ones are {}'.format(loss, list(self.valid_losses.keys()))
        self.tmpdir = None
        self.svmperf_learn = join(svmperf_base, 'svm_perf_learn')
        self.svmperf_classify = join(svmperf_base, 'svm_perf_classify')
        self.verbose=verbose
        self.loss = '-w 3 -l ' + str(self.valid_losses[loss])
        self.set_c(C)
        self.default_parameters = {'C':C} #compatibility with other quantifiers
        self.__name__ = 'SVMperf-'+loss

    def set_c(self, C):
        self.param_C = '-c ' + str(C)

    def set_params(self, **parameters):
        assert list(parameters.keys()) == ['C'], 'currently, only the C parameter is supported'
        self.set_c(parameters['C'])

    def fit(self, X, y):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model = join(self.tmpdir.name, 'model')
        traindat = join(self.tmpdir.name, 'train.dat')

        dump_svmlight_file(X, y, traindat, zero_based=False)

        cmd = ' '.join([self.svmperf_learn, self.param_C, self.loss, traindat, self.model])
        self.print('[Running]',cmd)
        p = subprocess.run(cmd.split(), stdout=PIPE, stderr=STDOUT)

        self.print(p.stdout.decode('utf-8'))


    def predict(self, X, y=None):
        assert self.tmpdir is not None, 'predict called before fit, or model directory corrupted'
        assert exists(self.model), 'model not found'
        if y is None:
            y = np.zeros(X.shape[0])

        random_code = '-'.join(str(randint(0,1000000)) for _ in range(5)) #this would allow to run parallel instances of predict
        predictions = join(self.tmpdir.name, 'predictions'+random_code+'.dat')
        testdat = join(self.tmpdir.name, 'test'+random_code+'.dat')
        dump_svmlight_file(X, y, testdat, zero_based=False)

        cmd = ' '.join([self.svmperf_classify, testdat, self.model, predictions])
        self.print('[Running]', cmd)
        p = subprocess.run(cmd.split(), stdout=PIPE, stderr=STDOUT)

        self.print(p.stdout.decode('utf-8'))

        estim_prevalence = (np.loadtxt(predictions) > 0).mean()

        return estim_prevalence


if __name__ == '__main__':


    Q = EM_Quantifier(probabilistic_learner=MultinomialNB, alpha=1)
    #Q = EM_Quantifier(probabilistic_learner=SVC, verbose=False, probability=True)
    #Q = SVMperfQuantifier(svmperf_base=SVMPERF_BASE, C=100, loss='nkld')
    print(Q.__name__)

    from data.rewiews_builder import ReviewsDataset
    (Xtr, ytr), (Xva, yva), (Xte, yte) = \
        loadDataset(dataset='imdb', mode='matrix', datapath='../../datasets/build/single')

    ytr = np.squeeze(ytr)
    yva = np.squeeze(yva)
    yte = np.squeeze(yte)

    Q.optim(Xtr, ytr, Xva, yva,
            error=mse,
            n_samples=100,
            sample_size=100,
            prev_range=define_prev_range(with_limits=True),
            refit=False,
            parameters = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
                          'fit_prior':[True,False]})
            #parameters={'C': [1e1, 1e2, 1e3, 1e4]})

    Q.fit(Xtr, ytr)

    sample_size = 500
    Xva_pos, Xva_neg, _, _ = split_pos_neg(Xva, yva)
    true_prevs = define_prev_range(True)
    estim_p = []
    for X,y,p in [sample_data(Xva_pos, Xva_neg, p, sample_size) for p in true_prevs]:
        p_ = Q.predict(X, y)
        print('true {:.4f}:\tq_={:.4f}'.format(p,p_))
        estim_p.append(p_)
    print('MSE: {}'.format(mse(true_prevs, estim_p)))
    print('MAE: {}'.format(mae(true_prevs, estim_p)))
    print('MRAE: {}'.format(mrae(true_prevs, estim_p)))
    print('MNKLD: {}'.format(mnkld(true_prevs, estim_p)))
