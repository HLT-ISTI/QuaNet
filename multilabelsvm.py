from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import sys

class MLSVC:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        #self.verbose = False if 'verbose' not in self.kwargs else self.kwargs['verbose']
        self.verbose = True

    def fit(self, X, y, **grid_search_params):
        assert len(y.shape)==2 and set(np.unique(y).tolist()) == {0,1}, 'data format is not multi-label'
        nD,nC = y.shape
        prevalence = np.sum(y, axis=0)
        self.svms = np.array([SVC(*self.args, **self.kwargs) for _ in range(nC)])
        if grid_search_params:
            self._print('grid_search activated with: {}'.format(grid_search_params))
            self.svms = [GridSearchCV(svm_i, refit=True, **grid_search_params) for svm_i in self.svms]
            # grid search fails if the category prevalence is less than parameter cv, in those cases we
            # simply place a svm (instead of a gridsearchcv)
            cv = 3 if 'cv' not in grid_search_params else grid_search_params['cv']
            if isinstance(cv, int):
                for i in np.argwhere(prevalence < cv).flatten():
                    self.svms[i] = self.svms[i].estimator
        for i in np.argwhere(prevalence==0):
            self.svms[i] = TrivialRejector()
        for c in range(nC):
            self._print('fit {}/{}'.format(c+1,nC))
            try:
                self.svms[c].fit(X,y[:,c])
            except Exception as e:
                print('error in ',c,X,y[:,c])
                if isinstance(self.svms[c],GridSearchCV):
                    self.svms[c] = self.svms[c].estimator
                    self.svms[c].fit(X, y[:, c])
                else: raise e
            if isinstance(self.svms[c], GridSearchCV):
                self._print('best: {}'.format(self.svms[c].best_params_))

    def predict(self, X):
        return np.vstack(map(lambda svmi:svmi.predict(X),self.svms)).T

    def _print(self, msg):
        if self.verbose>0:
            print(msg)

class TrivialRejector:
    def fit(self,*args,**kwargs): pass
    def predict(self, X): return np.zeros(X.shape[0])


from sklearn.datasets import make_multilabel_classification
svm = MLSVC()
X, y = make_multilabel_classification(n_samples=500, n_features=100, n_classes=10, n_labels=5, length=20,
                                      allow_unlabeled=False, random_state=1, sparse=True)
print(np.sum(y,axis=0))
#svm.fit(X,y , param_grid={'C':[1,10,100,1000]}, n_jobs=-1)
svm.fit(X,y)
Y_ = svm.predict(X)
from sklearn.metrics import f1_score
print(f1_score(y,Y_, average='macro'))
print(f1_score(y,Y_, average='micro'))


"""
w/o optimization
macroF1 = 0.516107802595
microF1 = 0.737886302938

optimized
macroF1 = 0.543074006681
microF1 = 0.769108280255
"""