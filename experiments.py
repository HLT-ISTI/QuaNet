import sys
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import scipy
from classification.multilabelsvm import MLSVC
from data.dataset_loader import TextCollectionLoader
from sklearn.model_selection import GridSearchCV
from util import disable_sklearn_warnings
from util.metrics import *
from sklearn.datasets import make_multilabel_classification
from time import time
from sklearn.multiclass import OneVsRestClassifier
import itertools
import os

evaluation_metrics = [macroF1, microF1, macroK, microK]

def print_evals(method, evals):
    print('%s:\t%s' % (method,'\t'.join(['%.3f' % e for e in evals])))

class Results:
    def __init__(self, outfile):
        if os.path.exists(outfile):
            self.previous_results = open(outfile, 'r').read()
        else:
            self.previous_results = ''
            with open(outfile, 'w') as out:
                out.write('\t'.join(['Learner', 'dataset', 'weight', 'feat_sel', 'norm', 'balanced'] + [f.__name__ for f in evaluation_metrics]+['time','optimC'])+'\n')

    def write_result(self, learner, dataset, cat_sel, weight, sublinear_tf, feat_sel, norm, balanced, metrics, time, optimC):
        id = self._compose_id(learner, dataset, cat_sel, weight, sublinear_tf, feat_sel, norm, balanced)
        metrics = '\t'.join(['%.4f'%metric_i for metric_i in metrics])
        time = '%.1f'%time
        if not isinstance(optimC, list):
            optimC = [optimC]
        for p in optimC: self.__assure_only_c(p)
        optimC = ' '.join([str(list(op.values())[0]) if op else '1' for op in optimC])
        with open(outfile, 'a') as out:
            line = '\t'.join([id, metrics, time, optimC])
            out.write(line+'\n')
            print(line)

    def __assure_only_c(self, params):
        if isinstance(params, dict) and len(params) > 1:
            raise ValueError('more parameters than simply C')

    def already_computed(self, learner, dataset, cat_sel, weight, sublinear_tf, feat_sel, norm, balanced):
        id = self._compose_id(learner, dataset, cat_sel, weight, sublinear_tf, feat_sel, norm, balanced)
        return id in self.previous_results

    def _compose_id(self, learner, dataset, cat_sel, weight, sublinear_tf, feat_sel, norm, balanced):
        dataset = dataset+'@'+(str(cat_sel) if cat_sel!=-1 else 'ALL')
        weight = ('log_' if sublinear_tf else '') + weight
        feat_sel = str(feat_sel) if feat_sel!=-1 else 'ALL'
        balanced = 'yes' if balanced else 'no'
        return '\t'.join([learner, dataset, weight, feat_sel, norm, balanced])


if __name__ == '__main__':

    n_jobs = 8
    c_range = [0.1, 0.9, 1, 1.1, 10, 100, 1000]
    cv = 5
    svmlearner = LinearSVC


    dataset = 'reuters21578'
    #configurations= [[True,False],[1000,0.1,None],[-1,90,10],['tfidf'], ['l2', 'l1'], [None, 'balanced']]
    configurations = [[True, False], [1000, 0.1, None], [-1], ['tfidf'], ['l2', 'l1'], [None, 'balanced']]
    configurations = list(itertools.product(*configurations))

    outfile='./comparative.csv'
    results = Results(outfile)

    for dataset in ['ohsumed']:
        for i,config in enumerate(configurations):
            print('completed {}/{}'.format(i,len(configurations)))
            #print('loading data', dataset)
            sublinear_tf, feat_sel, top_categories, vectorizer, norm, class_weight = config
            mlsvmcomputed = results.already_computed(MLSVC.__name__, dataset, top_categories, vectorizer, sublinear_tf, feat_sel, norm, class_weight)
            svmcomputed = results.already_computed(svmlearner.__name__, dataset, top_categories, vectorizer, sublinear_tf, feat_sel, norm, class_weight)

            if mlsvmcomputed and svmcomputed: continue

            data = TextCollectionLoader(dataset=dataset, vectorizer=vectorizer, sublinear_tf=sublinear_tf, feat_sel=feat_sel, top_categories=top_categories, norm=norm)
            Xtr, ytr = data.get_devel_set()
            Xte, yte = data.get_test_set()
            Xtr.sort_indices()
            Xte.sort_indices()

            # l1 = scipy.sparse.linalg.norm(Xtr, ord=1, axis=1)
            # print(np.mean(l1), np.std(l1))
            # continue

            if not mlsvmcomputed:
                mlsvm = MLSVC(verbose=False, n_jobs=n_jobs, estimator=svmlearner, class_weight=class_weight)
                mlsvm.fit(Xtr, ytr, param_grid={'C':c_range}, cv=cv)
                training_time = mlsvm.training_time
                Y_ = mlsvm.predict(Xte)
                metrics = [metric(yte, Y_) for metric in evaluation_metrics]
                results.write_result(MLSVC.__name__, dataset, top_categories, vectorizer, sublinear_tf, feat_sel, norm, class_weight, metrics, training_time, mlsvm.best_params())

            if not svmcomputed:
                svm = GridSearchCV(OneVsRestClassifier(svmlearner(class_weight=class_weight), n_jobs=n_jobs),
                                   refit=True, param_grid={'estimator__C':c_range}, cv=cv)
                #try:
                time_ini = time()
                svm.fit(Xtr,ytr)
                training_time = time() - time_ini
                Y_ = svm.predict(Xte)
                metrics = [metric(yte, Y_) for metric in evaluation_metrics]
                results.write_result(svmlearner.__name__, dataset, top_categories, vectorizer, sublinear_tf, feat_sel, norm,
                                     class_weight, metrics, training_time, svm.best_params_)
                # except Exception as e:
                #     print(e)
                #     print('error in fit')


