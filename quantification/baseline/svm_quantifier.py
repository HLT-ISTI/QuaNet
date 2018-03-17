import util.disable_sklearn_warnings
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix, issparse
import numpy as np

class SVMclassifyAndCount:
    def __init__(self, C_range = [1, 10, 100, 1000], **kwargs):
        self.svm = GridSearchCV(OneVsRestClassifier(LinearSVC(**kwargs), n_jobs=-1),
                       param_grid={'estimator__C': C_range}, refit=True)

    def fit(self, X, y):
        print('training svm...')
        X = self.prepare_matrix(X)
        self.svm.fit(X, y)
        return self

    def predict_prevalences(self, X):
        X = self.prepare_matrix(X)
        y_ = self.svm.predict(X)
        return np.mean(y_,axis=0)

    def prepare_matrix(self, X):
        if not issparse(X):
            X = csr_matrix(X)
            X.sort_indices()
        return X

