import cPickle as pickle
import os

import numpy as np
from scipy.sparse import csr_matrix, vstack

from utils.helpers import create_if_not_exists


class WeightedVectors:
    def __init__(self, vectorizer, from_dataset, from_category, trX, trY, vaX, vaY, teX, teY, run_params_dic=None):
        self.name = from_dataset
        self.positive_cat = from_category
        self.vectorizer_name = vectorizer
        self.trX = csr_matrix(trX)
        self.trY = trY
        self.vaX = csr_matrix(vaX)
        self.vaY = vaY
        self.teX = csr_matrix(teX)
        self.teY = teY
        self.run_params_dic = run_params_dic

    def pickle(self, outdir, outfile_name):
        create_if_not_exists(outdir)
        pickle.dump(self, open(os.path.join(outdir,outfile_name), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def unpickle(indir, infile_name):
        return pickle.load(open(os.path.join(indir,infile_name), 'rb'))

    def get_train_set(self):
        return self.trX, self.trY

    def get_validation_set(self):
        return self.vaX, self.vaY

    def get_devel_set(self):
        return vstack((self.trX,self.vaX), format='csr'), np.concatenate((self.trY,self.vaY))

    def get_test_set(self):
        return self.teX, self.teY

    def num_features(self):
        return self.trX.shape[1]

    def num_categories(self):
        return self.teY.shape[1]

    def num_devel_documents(self):
        return self.num_tr_documents()+self.num_val_documents()

    def num_tr_documents(self):
        return len(self.trY)

    def num_val_documents(self):
        return len(self.vaY)

    def num_test_documents(self):
        return len(self.teY)

    def get_categories(self):
        return ['negative','positive']

    def get_learning_parameters(self):
        return self.run_params_dic


