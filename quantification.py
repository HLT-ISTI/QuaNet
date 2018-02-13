import disable_sklearn_warnings
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from torch.autograd import Variable
import numpy as np
from data.dataset_loader import TextCollectionLoader
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix, issparse

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_p=0.5):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        # self.fc3 = nn.Linear(int(hidden_size/2), num_classes)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout_p = dropout_p

    #x is a text collection (shape=(nD,nF)); in terms of quantification is just one example therefore
    def forward(self, x):
        x = torch.mm(x.t(), x).view(1, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_p, training=self.training)
        x = self.fc2(x)
        if self.training == False:
            x = torch.clamp(x, 0, 1)
        #x = F.sigmoid(x)
        # out = F.relu(out)
        # out = F.dropout(out, self.dropout_p, training=self.training)
        # out = self.fc3(out)
        return x.view(1, -1)

def variable_from_numpy(numpy):
    var = Variable(torch.from_numpy(numpy).float(), requires_grad=False)
    return var.cuda() if args.cuda else var

def prepare_variables(X, y):
    nD = X.shape[0]
    X = variable_from_numpy(X.toarray() if issparse(X) else X)
    y = variable_from_numpy(np.sum(y, axis=0) / nD).view(1, -1)

    return X,y

def sample_collection(X, y=None, min_size=100):
    nD = X.shape[0]
    doc_indexes = np.arange(nD)
    np.random.shuffle(doc_indexes)
    n = np.random.randint(min_size, nD)
    Xsample = X[doc_indexes[:n]]
    if y is not None:
        ysample = y[doc_indexes[:n]]
        return Xsample, ysample
    else:
        return Xsample


# Training the Model
def train(X, y, net, evaluation_measure=None, num_steps = 10000, loss_ave_steps = 100, test_steps=1000, learning_rate = 0.001, weight_decay = 0.0001, Xte=None, yte=None):
    net.train(mode=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_ave = 0
    for i in range(1, num_steps+1):
        Xs, ys = sample_collection(X, y)
        Xs, ys = prepare_variables(Xs, ys)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        estim_prev = net(Xs)
        loss = criterion(estim_prev, ys)
        loss.backward()
        optimizer.step()

        loss_ave += loss.data[0]
        if i % loss_ave_steps == 0:
            print('Step: [%d/%d], Loss: %.12f' % (i, num_steps, loss_ave / loss_ave_steps))
            loss_ave = 0

        if evaluation_measure is not None and i % test_steps == 0:
            test(Xte, yte, net, evaluation_measure=evaluation_measure)


# Test the Model
def test(X, y, net, evaluation_measure, verbose=True):
    net.train(mode=False)

    X, true_prevalences = prepare_variables(X, y)

    estimated_prevalences = net(X)

    mae = evaluation_measure(estimated_prevalences, true_prevalences)

    if verbose:
        print('estimated:', estimated_prevalences)
        print('true:', true_prevalences)
        print('Net-MAE %.8f' % mae[0])

    return mae[0]

# ---------------------------------------------------------------
# SVM routines
# ---------------------------------------------------------------
def train_svm(X, y):
    print('training svm...')

    if not issparse(X):
        X = csr_matrix(X)
        X.sort_indices()

    svm = GridSearchCV(OneVsRestClassifier(LinearSVC(class_weight='balanced'), n_jobs=-1),
                       param_grid={'estimator__C': [1, 10, 100, 1000]}, refit=True)

    return svm.fit(X, y)

def test_svm(X, y, svm, evaluation_measure, verbose=True):
    if not issparse(X):
        X = csr_matrix(X)
        X.sort_indices()

    y_ = svm.predict(X)

    nD = X.shape[0]
    estimated_prevalences = variable_from_numpy(np.sum(y_, axis=0) / nD)
    true_prevalences = variable_from_numpy(np.sum(y, axis=0) / nD)

    mae = evaluation_measure(estimated_prevalences, true_prevalences)

    if verbose:
        print('estimated:', estimated_prevalences)
        print('true:', true_prevalences)
        print('SVM-MAE %.8f' % mae[0])

    return mae[0]

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
# helper to parse the code for a dataset: <collection>[@<cats>][F<feats>][W<[log]weight>], e.g.: reuters21578@90F1000Wlogtfidf
def parse_dataset_code(dataset_code):
    import re
    def try_get(regex, default):
        try: return re.search(regex, dataset_code).group(1)
        except: return default
    dataset = [d for d in valid_datasets if dataset_code.startswith(d)]
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
    print('loading ' + args.dataset)
    data = TextCollectionLoader(dataset=dataset, vectorizer=weight, sublinear_tf=log, feat_sel=features, rep_mode='dense', top_categories=categories)

    nF = data.num_features()
    nC = data.num_categories()

    net = Net(input_size=nF * nF, hidden_size=args.hidden, num_classes=nC)
    net = net.cuda() if args.cuda else net

    # Evaluation measures
    mean_absolute_error = nn.L1Loss()

    Xtr, ytr = data.get_devel_set()
    Xte, yte = data.get_test_set()

    train(Xtr, ytr, net, evaluation_measure=mean_absolute_error, Xte=Xte, yte=yte, num_steps=args.iter, learning_rate=args.lr, weight_decay=args.weight_decay)

    mae_net = test(Xte, yte, net, evaluation_measure=mean_absolute_error)

    svm = train_svm(Xtr, ytr)
    mae_svm = test_svm(Xte, yte, svm, evaluation_measure=mean_absolute_error)

    tr_prev = torch.from_numpy(np.sum(ytr, axis=0))
    te_prev = torch.from_numpy(np.sum(yte, axis=0))
    mae_naive = mean_absolute_error(tr_prev, te_prev)[0]

    print('Net-MAE:\t%.4f' % mae_net)
    print('SVM-MAE:\t%.4f' % mae_svm)
    print('Naive-MAE:\t%.4f' % mae_naive)



# TODO: random indexing or projection
# TODO: evaluation trials-policies
# TODO: evaluation metrics
# TODO: batch? loss-plot?
# TODO: sampling policies
# TODO: KLD loss?
# TODO: normalize the FxF matrix?
# TODO: baseline that outputs the training prevalences
if __name__ == '__main__':
    valid_datasets = ['reuters21578']
    parser = argparse.ArgumentParser(description='Text Quantification Net')
    parser.add_argument('--iter', type=int, default=10000, metavar='I',
                        help='number of iterations to train (default: 10000)')
    parser.add_argument('--hidden', type=int, default=512, metavar='H',
                        help='hidden size (default: 512)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W',
                        help='weight decay (default: 1e-4)') #previously: 1e-4
    parser.add_argument('--dataset', '--d', default='reuters21578@115F500Wlogtfidf', type=str, metavar='D',
                        help='dataset to load in {} (default: {})'.format(', '.join(valid_datasets), valid_datasets[0]))

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('CUDA:', args.cuda)
    main(args)




