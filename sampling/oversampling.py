import numpy as np
from scipy.sparse import csr_matrix

from feature_selection.tsr_function import get_supervised_matrix, information_gain, get_tsr_matrix


class DRO:

    def __init__(self, upsampling=0.5, downsampling=0.):
        if downsampling!=0: raise NotImplementedError('Downsampling not yet implemented!')
        self.upsampling = upsampling
        self.downsampling = downsampling

    def fit(self, X, y):
        assert isinstance(y, np.ndarray), 'wrong format, np.ndarray expected by {} found'.format(type(y))
        y = y.reshape(-1)
        assert y.ndim==1, 'only binary problems accepted'
        assert y.shape[0] == X.shape[0]

        self.tsr_array = get_tsr_matrix(get_supervised_matrix(X, y), information_gain) # shape (1,nF)
        self.Z = X.dot(X.T * self.tsr_array)
        self.y = y

        return self

    def transform(self, X, y=None):
        assert hasattr(self, 'Z'), 'transform called before fit'
        assert y is None, 'y shall only be passed to fit'
        y = self.y
        Z = self.Z

        npos = y[y == 1].size
        nneg = y.size - npos

        # number of times each positive example has to be oversampled
        alpha = self.upsampling * nneg / (npos - self.upsampling * npos)

        nD = nneg + npos*alpha
        nF = X.shape[1]
        nL = self.Z.shape[1]
        O = csr_matrix(shape=(nD, nF + nL))
        Oy = np.zeros(nD)

        offset = 0
        for i,di in enumerate(X):
            nsamples = 1 if y[i]==0 else alpha
            for _ in range(nsamples):
                O[offset, 0:nF] = di
                O[offset, nF:]  = self._sample(Z[i])
                Oy[offset] = y[i]
                offset += 1

    def fit_transform(self, X, y):
        return self.fit(X,y).transform(X)

    def _sample(self,Zi):
        pass

