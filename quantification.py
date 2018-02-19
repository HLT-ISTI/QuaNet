import util.disable_sklearn_warnings
import argparse, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from torch.autograd import Variable
import numpy as np
from data.dataset_loader import TextCollectionLoader
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix, issparse


np.random.seed(1)
torch.manual_seed(1)

# TODO: adjust_learningrate (doesn't seem to help)
# TODO: inspect the (Xt x X)/n thing (doesn't seem to help in the long term; the loss is much lower in the initial steps though)
# TODO: progress bar!
# TODO: random indexing or projection
# TODO: evaluation trials-policies
# TODO: evaluation metrics
# TODO: batch? or cumulate gradients...? loss-plot?
# TODO: sampling policies
# TODO: KLD loss?
# TODO: normalize the FxF matrix?
# TODO: add layers; does not seem to help
# TODO: use, as an evaluation metric, the p-values of two proportions in the Z-distribution (for each category);
# ...and figure out how to sum them up. This should not be difficult because all them share the number of documents (population size)

def variable_from_numpy(numpy):
    var = Variable(torch.from_numpy(numpy).float())
    return var.cuda() if args.cuda else var

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_p=0.5, normalize=False):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout_p = dropout_p
        self.normalize = normalize

    #x is a text collection (shape=(nD,nF)); in terms of quantification is just one example therefore
    def forward(self, X):
        nD,nF = X.shape
        f = torch.mm(X.t(), X)
        if self.normalize:
            f /= nD
        triangular_mask = (torch.triu(torch.ones(nF, nF)) == 1)
        triangular_mask = triangular_mask.cuda() if args.cuda else triangular_mask
        f = f[triangular_mask] # upper-triangular values
        f = f.view(1, -1)
        out = self.fc1(f)
        out = F.relu(out)
        out = F.dropout(out, self.dropout_p, training=self.training)
        out = self.fc2(out)
        out = F.relu(out)
        out = F.dropout(out, self.dropout_p, training=self.training)
        out = self.fc3(out)
        if self.training == False:
            out = torch.clamp(out, 0, 1)
        return out

    def train_step(self, X, y, optimizer, criterion):
        self.train()
        X = self.prepare_matrix(X)
        true_prev = self.classif2prev(y)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        estim_prev = self.forward(X)
        loss = criterion(estim_prev, true_prev)
        loss.backward()
        optimizer.step()
        return loss.data[0]

    def prepare_matrix(self, X):
        return variable_from_numpy(X.toarray() if issparse(X) else X)

    def classif2prev(self, y):
        return variable_from_numpy(count(y)).view(1, -1)

    def predict_prevalences(self, X):
        self.eval()
        X = self.prepare_matrix(X)
        return self.forward(X)

    # Test the Model
    def evaluation(self, X, y, evaluation_measure, verbose=False):
        estim_prev = self.predict_prevalences(X)
        true_prev = self.classif2prev(y)
        mae = evaluation_measure(estim_prev, true_prev)

        if verbose:
            print('estimated:', estim_prev)
            print('true:', true_prev)
        print('Net-MAE %.8f' % mae[0])

        mae = mae[0].data
        mae = mae.cpu() if args.cuda else mae
        return mae.numpy()[0]

def count(y):
    return (np.sum(y, axis=0) / y.shape[0])

def sample_collection(X, y=None):
    nD = X.shape[0]
    n = np.random.randint(int(0.1*nD), nD)
    doc_indexes = np.random.permutation(nD)[:n]
    Xsample = X[doc_indexes]
    if y is not None:
        ysample = y[doc_indexes]
        return Xsample, ysample
    else:
        return Xsample

def adjust_learning_rate(optimizer, iter, each):
    # sets the learning rate to the initial LR decayed by 0.1 every 'each' iterations
    lr = args.lr * (0.1 ** (iter // each))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr
    return lr

# Training the Model
def train(X, y, net, evaluation_measure=None, num_steps = 50000, loss_ave_steps = 10, test_steps=1000, learning_rate = 0.001, weight_decay = 0.0001, Xte=None, yte=None):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_ave = 0
    for i in range(1, num_steps+1):
        lr = adjust_learning_rate(optimizer, i, 10000)
        #lr = args.lr
        Xs, ys = sample_collection(X, y)

        loss = net.train_step(Xs, ys, optimizer, criterion)

        loss_ave += loss
        if i % loss_ave_steps == 0:
            print('Step: [%d/%d, lr=%f], Loss: %.12f' % (i, num_steps, lr, loss_ave / loss_ave_steps))
            loss_ave = 0

        if evaluation_measure is not None and i % test_steps == 0:
            net.evaluation(Xte, yte, evaluation_measure=evaluation_measure)



# ---------------------------------------------------------------
# SVM routines
# ---------------------------------------------------------------
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
        return count(y_)

    def prepare_matrix(self, X):
        if not issparse(X):
            X = csr_matrix(X)
            X.sort_indices()
        return X

    def evaluation(self, X, y, evaluation_measure, verbose=False):
        estimated_prevalences = self.predict_prevalences(X)
        estimated_prevalences = variable_from_numpy(estimated_prevalences)

        true_prevalences = count(y)
        true_prevalences = variable_from_numpy(true_prevalences)

        mae = evaluation_measure(estimated_prevalences, true_prevalences)

        if verbose:
            print('estimated:', estimated_prevalences)
            print('true:', true_prevalences)
            print('SVM-MAE %.8f' % mae[0])

        mae = mae[0].data
        mae = mae.cpu() if args.cuda else mae
        return mae.numpy()[0]

def wilcoxon_comparison(X, y, method1, method2, eval_metric, lower_is_better=True, until_p = 0.05, max_iter=100):
    from scipy.stats import wilcoxon
    signficance_detected = False
    results1, results2 = [], []
    while not signficance_detected:
        Xs, ys = sample_collection(X,y)

        results1.append(method1.evaluation(Xs, ys, eval_metric))
        results2.append(method2.evaluation(Xs, ys, eval_metric))

        if len(results1) > 30:
            _,p = wilcoxon(results1, results2)
            signficance_detected = (p < until_p)
            print(results1[-1], results2[-1], signficance_detected)
        else:
            print(results1[-1],results2[-1])
        if len(results1) > max_iter:
            break

    if signficance_detected:
        r1ave = np.mean(results1)
        r2ave = np.mean(results2)
        lower,bigger = (method1,method2) if r1ave < r2ave else (method2,method1)
        return lower if lower_is_better else bigger

    print('Wilcoxon indetermined after {} iterations'.format(max_iter))
    return None


def random_split_evaluation(X, y, methods, eval_metrics, nsplits=10, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    nD = y.shape[0]
    doc_indexes = np.random.permutation(nD)
    results = np.zeros((nsplits, len(methods), len((eval_metrics))), dtype=float)
    for s in range(nsplits):
        split_indexes = doc_indexes[s*nD//nsplits:(s+1)*nD//nsplits]
        Xs, ys = X[split_indexes], y[split_indexes]

        for i, method_i in enumerate(methods):
            for j,eval_j in enumerate(eval_metrics):
                results[s,i,j] = method_i.evaluation(Xs, ys, eval_j)

    return np.mean(results, axis=0), np.std(results, axis=0)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
# helper to parse the code for a dataset: <collection>[@<cats>][F<feats>][W<[log]weight>], e.g.: reuters21578@90F1000Wlogtfidf
def parse_dataset_code(dataset_code):
    import re
    def try_get(regex, default):
        try: return re.search(regex, dataset_code).group(1)
        except: return default
    dataset = [d for d in TextCollectionLoader.valid_datasets if dataset_code.startswith(d)]
    if len(dataset)!=1: raise ValueError('unknown dataset code')
    else:
        dataset = dataset[0]
    categories = int(try_get('.*@(\d+).*', -1))
    features = int(try_get('.*F(\d+).*', None))
    weight = try_get('.*W([a-z]+).*', 'logtfidf')
    if weight.startswith('log'):
        weight=weight[3:]
        log = True
    else:
        log = False
    return dataset, categories, features, weight, log

def main(args):
    dataset, categories, features, weight, log = parse_dataset_code(args.dataset)
    print('loading ' + dataset, categories, features, weight, log)
    data = TextCollectionLoader(dataset=dataset, vectorizer=weight, sublinear_tf=log, feat_sel=features, rep_mode='dense', top_categories=categories)

    nF = data.num_features()
    nC = data.num_categories()

    net = Net(input_size=int((nF+1)*nF/2), hidden_size=args.hidden, num_classes=nC)
    net = net.cuda() if args.cuda else net

    # Evaluation measures
    mean_absolute_error = nn.L1Loss()

    Xtr, ytr = data.get_devel_set()
    Xte, yte = data.get_test_set()

    tr_prev = variable_from_numpy(np.sum(ytr, axis=0) / ytr.shape[0])
    te_prev = variable_from_numpy(np.sum(yte, axis=0) / yte.shape[0])
    mae_naive = mean_absolute_error(tr_prev, te_prev)[0]
    print('train_prevalence:', tr_prev)
    print('test_prevalence:', te_prev)
    print('Naive-MAE:\t%.8f' % mae_naive)
    #sys.exit()

    train(Xtr, ytr, net, evaluation_measure=mean_absolute_error, Xte=Xte, yte=yte, num_steps=args.iter, learning_rate=args.lr, weight_decay=args.weight_decay)
    mae_net = net.evaluation(Xte, yte, evaluation_measure=mean_absolute_error, verbose=True)

    svm = SVMclassifyAndCount(class_weight='balanced')
    svm.fit(Xtr, ytr)
    mae_svm = svm.evaluation(Xte, yte, evaluation_measure=mean_absolute_error, verbose=True)

    winner = wilcoxon_comparison(Xte, yte, net, svm, eval_metric=mean_absolute_error, lower_is_better=True)
    if winner:
        print('Wilcoxon test, winner: ' + winner.__class__.__name__)

    print('Net-MAE:\t%.8f' % mae_net)
    print('SVM-MAE:\t%.8f' % mae_svm)
    print('Naive-MAE:\t%.8f' % mae_naive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Quantification Net')
    parser.add_argument('--iter', type=int, default=10000, metavar='I',
                        help='number of iterations to train (default: 10000)')
    parser.add_argument('--hidden', type=int, default=512, metavar='H',
                        help='hidden size (default: 512)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--dataset', '--d', default='reuters21578@115F500Wlogtfidf', type=str, metavar='D',
                        help='dataset to load (default: reuters21578@115F500Wlogtfidf)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('CUDA:', args.cuda)
    main(args)




