from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import sys
from sklearn.externals.joblib import Parallel, delayed

from data.dataset_loader import TextCollectionLoader


class SLSVC:
    """
    Single-Label Support Vector Machine, with category gaps
    """

    def __init__(self, estimator=LinearSVC, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.verbose = False if 'verbose' not in self.kwargs else self.kwargs['verbose']
        self.estimator = estimator(*args, **kwargs)

    def fit(self, X, y, classes=None):
        self._fit_y_shape = len(y.shape)
        if classes is None and len(y.shape)==2:
            classes = np.array(range(y.shape[1]))
        if isinstance(classes, list):
            classes = np.array(classes)
        self._check_consistency(X, y, classes)
        if len(y.shape)==2:
            y = self._adapt_class_matrix2array(y, classes)
        self.classes = classes
        self.estimator.fit(X=X,y=y)
        return self

    def predict(self, X):
        y_ = self.estimator.predict(X)
        print(y_)
        if self._fit_y_shape == 2:
            y_ = self._adapt_class_array2matrix(y_, self.classes)
        return y_

    def predict_proba(self, X):
        prob = self.estimator.predict_proba(X)
        return self.__reorder_output(prob, self.estimator.classes_, self.classes)

    def decision_function(self, X):
        raise ValueError('not working if the problem was binary, TODO: adapt to a matrix output if needed')
        decisions = self.estimator.decision_function(X)
        return self.__reorder_output(decisions, self.estimator.classes_, self.classes)

    def __reorder_output(self, estimator_output, estimator_classes, self_classes):
        if len(estimator_classes) != len(self_classes) or not np.all(estimator_classes == self_classes):
            output_reorderd = np.zeros((X.shape[0], len(self_classes)), dtype=float)
            for from_c, class_label in enumerate(estimator_classes):
                output_reorderd[:, self_classes == class_label] = estimator_output[:, from_c]
            return output_reorderd
        else:
            return estimator_output

    def _adapt_class_matrix2array(self, y, classes):
        nD,nC = y.shape
        y_ = np.zeros(nD, dtype=classes.dtype)
        for c in range(nC):
            label = (classes[c] if classes is not None else c)
            y_[y[:,c]==1]=label
        return y_

    def _adapt_class_array2matrix(self, y, classes):
        nD,nC = len(y), len(classes)
        y_ = np.zeros((nD,nC), dtype=int)
        for i,label in enumerate(y):
            y_[i,classes==label] = 1
        return y_

    def _check_consistency(self, X, y, classes):
        nD = X.shape[0]
        if len(y.shape)==2:
            if y.shape[0] != nD:
                raise ValueError('different dimensions found for X and y')
            if y.shape[1] != len(classes):
                raise ValueError('different dimensions found for y and the number of classes')
            if set(np.unique(y).tolist())!={0,1}:
                raise ValueError('the matrix is not binary: a conversion is not possible')
            if not np.all(np.sum(y,axis=1)==1):
                raise ValueError('not all documents are labeled with exactly one label')
        elif len(y.shape)==1:
            if classes is not None:
                if not set(np.unique(y).tolist()).issubset(np.unique(classes).tolist()):
                    raise ValueError('y contains labels which are outside the scope of classes')

if __name__=='__main__':
    #mlb = MultiLabelBinarizer()
    X = np.random.rand(8,100)
    #y = np.array([[1,0,0],[1,0,0],[0,0,1],[0,0,1],[1,0,0],[1,0,0],[0,0,1],[0,0,1]])
    y = np.array([0, 0, 2, 2, 0, 0, 2, 2])

    #y = mlb.fit_transform(y)

    #classes = np.array(['Z', 'C', 'B', 'A'])
    #print(y,classes)
    svm = SLSVC(n_jobs=-1, estimator=SVC, probability=True)
    svm.fit(X,y)
    print(svm.predict(X))
    #y_ = svm.predict_proba(X)
    #print(y_)
    #y_ = svm.decision_function(X)
    #print(y_)