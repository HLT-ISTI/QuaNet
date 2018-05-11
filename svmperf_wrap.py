from os.path import join, exists
import subprocess
from subprocess import PIPE, STDOUT
import numpy as np
import tempfile
from sklearn.datasets import dump_svmlight_file
from data.dataset_loader import parse_dataset_code
import sys
from quantification.helpers import split_pos_neg, sample_data
from random import randint

class SVMperfQuantifier:

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

    def set_c(self, C):
        self.param_C = '-c ' + str(C)

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


    def optim(self, Xtr, ytr, Xva, yva, error,
              n_samples,
              sample_size,
              range_C=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
              prev_range=None, n_jobs=-1):
        from sklearn.externals.joblib import Parallel, delayed

        if prev_range is None:
            prev_range = 1 / 20 + np.arange(19) * 1 / 20  # [0.05, 0.1, 0.15, ... , 0.95]

        #create n_samples of sample_length
        Xva_pos, Xva_neg,_,_ = split_pos_neg(Xva,yva)
        samples_prevalences = np.repeat(prev_range, n_samples / prev_range.size)
        samples = [sample_data(Xva_pos, Xva_neg, p, sample_size) for p in samples_prevalences] # each sample is (X,y,real_p)
        true_prevs = [p for _,_,p in samples]

        verb = self.verbose
        self.verbose = False

        best_c, best_error = None, None
        for c in range_C:
            print('checking c={} ... '.format(c), end='')
            sys.stdout.flush()

            self.set_c(c)
            self.fit(Xtr,ytr)
            estim_prevs = Parallel(n_jobs=n_jobs)(delayed(self.predict)(X,y) for X,y,p in samples)
            score = error(true_prevs, estim_prevs)

            print('\tgot {} score {:.5f}'.format(error.__name__, score), end='')
            if best_error is None or score < best_error:
                best_error, best_c = score, c
                print('\t[best found]', end='')
            print()

        self.set_c(best_c)
        self.verbose = verb

    def print(self, msg, end='\n'):
        if self.verbose:
            print(msg,end=end)

if __name__ == '__main__':

    svmperf_base = '/home/moreo/svm-perf-quantification'

    svm = SVMperfQuantifier(svmperf_base=svmperf_base, C=100, loss='nkld')
    data = parse_dataset_code('reuters21578@1F1000Wtfidf')
    Xtr, ytr = data.get_train_set()
    Xva, yva = data.get_validation_set()
    Xte, yte = data.get_test_set()

    ytr = np.squeeze(ytr)
    yva = np.squeeze(yva)
    yte = np.squeeze(yte)

    svm.optim(Xtr, ytr, Xva, yva,
              range_C=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
              n_samples=100)

    svm.fit(Xtr, ytr)
    p = svm.predict(Xte, yte)

    print('true',np.mean(yte))
    print('estimated',p)