import argparse
import ntpath
from os import makedirs
from os.path import join, exists

import numpy as np
import torch.nn.functional as F
from scipy.sparse import issparse
from torch.autograd import Variable

from data.dataset_loader import TextCollectionLoader
from util.interactive_trainer import *

np.random.seed(1)
torch.manual_seed(1)

# done: loss modified WRONG, from MSE -> multilabelsoftmargin and binarycrossentropy (Im not classifying! but predicting prevalence!, shame!)
# TODO: singlelabel means the prevalences should sum up to 1!
# TODO: solve the issue of training, validation, and test
# TODO: inspect the (Xt x X)/n thing (doesn't seem to help in the long term; the loss is much lower in the initial steps though)
# TODO: progress bar!
# TODO: random indexing or projection
# TODO: loss-plot?
# TODO: sampling policies: current version -> check if it is merely learning the training prevalence
# TODO: KLD loss?
# done: add layers; does not seem to help
# TODO: random seed in params



def variable_from_numpy(numpy, astype=float):
    tensor = torch.from_numpy(numpy)
    if isinstance(astype,str):
        totype = getattr(tensor, astype)
    else:
        totype = getattr(tensor, astype.__name__)
    tensor = totype()
    var = Variable(tensor)
    return var.cuda() if args.cuda else var

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_p=0.5, normalize=False):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout_p = dropout_p
        self.normalize = normalize
        if args.cuda:
            self.cuda()
        #self.classmode = classmode

    #x is a text collection (shape=(nD,nF)); in terms of quantification is just one example therefore
    def forward(self, X):
        nD,nF = X.shape
        T = torch.mm(X.t(), X)
        if self.normalize:
            T /= nD
        triangular_mask = (torch.triu(torch.ones(nF, nF)) == 1)
        triangular_mask = triangular_mask.cuda() if args.cuda else triangular_mask
        T = T[triangular_mask].view(1, -1) # upper-triangular values

        out = self.fc1(T)
        out = F.relu(out)
        out = F.dropout(out, self.dropout_p, training=self.training)
        out = self.fc2(out)
        out = F.relu(out)
        out = F.dropout(out, self.dropout_p, training=self.training)
        out = self.fc3(out)
        # for some reason, this block doesn not work; for now, I'm going with the simple clamp in test
        # if self.classmode == 'singlelabel':
        #     out = nn.Softmax(out,dim=1)
        # else:
        #     #print(out)
        #     out = F.sigmoid(out)
        out = F.sigmoid(out)
        # if self.training == False:
        #     out = torch.clamp(out, 0, 1)
        return out

    def train_step(self, X, y, optimizer, criterion):
        self.train()
        X = self.prepare_matrix(X)
        true_prev = self.classif2prev(y)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        estim_prev = self.forward(X)
        loss = criterion(estim_prev.view(-1), true_prev.view(-1))
        loss.backward()
        optimizer.step()
        return loss.data[0]

    def prepare_matrix(self, X):
        return variable_from_numpy(X.toarray() if issparse(X) else X)

    def classif2prev(self, y):
        return variable_from_numpy(np.mean(y,axis=0)).view(1, -1)

    def predict_prevalences(self, X):
        self.eval()
        X = self.prepare_matrix(X)
        return self.forward(X)

    # Test the Model
    def validation(self, X, y, evaluation_measure):
        estim_prev = self.predict_prevalences(X)
        true_prev = self.classif2prev(y)
        mae = evaluation_measure(estim_prev, true_prev)

        print('True_prev',true_prev)
        print('estim_prev',estim_prev)
        print('Net-MAE %.8f' % (mae[0]))

        mae = mae[0].data
        mae = mae.cpu() if args.cuda else mae
        return mae.numpy()[0]

def sample_collection(X, y):
    nD = X.shape[0]
    n = np.random.randint(int(0.1*nD), nD)
    doc_indexes = np.random.permutation(nD)[:n]
    Xsample = X[doc_indexes]
    ysample = y[doc_indexes]
    return Xsample, ysample

# def sample_collection(X, y, prevalence=None):
#     nD, nC = y.shape
#     assert nC == 1, 'only supported for binary classification'
#     if prevalence is None:
#         prevalence = np.random.rand()
#
#     y_ = y.flatten()
#     Xpos, Xneg = X[y_ == 1], X[y_ != 1]
#
#     sample_length = np.random.randint(int(0.1*nD), nD)
#     sample_pos = int(sample_length * prevalence)
#     sample_neg = sample_length - sample_pos
#
#     Xsample = vstack((resample(Xpos, n_samples=sample_pos), resample(Xneg, n_samples=sample_neg)))
#     ysample = np.array([1]*sample_pos + [0]*sample_neg).reshape(-1,1)
#     rand_order = np.random.permutation(sample_pos+sample_neg)
#
#     return Xsample[rand_order], ysample[rand_order]

class LossHistory:
    def __init__(self, retain=10):
        self._hist = []
        self._retain = retain
        self.min_loss = None

    def commit(self, loss):
        self._hist.append(loss)
        if len(self._hist)>self._retain:
            self._hist=self._hist[1:]
        if self.min_loss is None or loss < self.min_loss:
            self.min_loss = loss

    def mean(self):
        return np.mean(self._hist)

def adjust_learning_rate(optimizer, iter, each):
    # sets the learning rate to the initial LR decayed by 0.1 every 'each' iterations
    lr = args.lr * (0.1 ** (iter // each))
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)
    return lr

def get_learning_rate(optimizer):
    for param_group in optimizer.state_dict()['param_groups']:
        return param_group['lr']

# Training the Model
def train(X, y, net, evaluation_measure=None, num_steps = 50000, loss_ave_steps = 100, test_steps=1000, learning_rate = 0.001, weight_decay = 0.0001, Xte=None, yte=None):
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    nD,nF = X.shape
    nD,nC = y.shape
    print('nF',nF)

    if args.interactive:
        innt = InteractiveNeuralTrainer()
        innt.add_interaction('w', increase_lr(optimizer, factor=10.))
        innt.add_interaction('s', decrease_lr(optimizer, factor=.1))
        innt.add_interaction('a', decrease_weight_decay(optimizer=optimizer, factor=.5))
        innt.add_interaction('d', increase_weight_decay(optimizer=optimizer, factor=2.))
        innt.add_interaction('r',
                             reboot(
            net, {'input_size':int((nF + 1) * nF / 2), 'hidden_size':args.hidden, 'num_classes':nC, 'normalize':True}, optimizer, tracked_optim_params=['lr','weight_decay']
                             ), synchronized=True)
        innt.add_interaction('v', lambda :net.validation(Xte, yte, evaluation_measure=evaluation_measure))
        innt.add_interaction('q', quick_save(net, 'checkpoint'), synchronized=True)
        innt.add_interaction('e', quick_load(net, 'checkpoint'), synchronized=True)
        innt.start()

    loss_history = LossHistory(retain=100)
    best_eval_loss = None
    for iter in range(1, num_steps+1):
        if args.interactive: innt.synchronize()

        #lr = adjust_learning_rate(optimizer, iter, 10000)
        lr = get_learning_rate(optimizer)
        Xs, ys = sample_collection(X, y)

        loss = net.train_step(Xs, ys, optimizer, criterion)

        loss_history.commit(loss)
        print('\rStep: [%d/%d, lr=%f], Loss: %.12f' % (iter, num_steps, lr, loss_history.mean()), end='')

        if evaluation_measure is not None and iter % test_steps == 0:
            eval_loss = net.validation(Xte, yte, evaluation_measure=evaluation_measure)
            for i in range(10):
                SXte, Syte = sample_collection(Xte, yte)
                net.validation(SXte, Syte, evaluation_measure=evaluation_measure)
            if best_eval_loss is None or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                model_name = join(args.modeldir, net.__class__.__name__ + '_' + ntpath.basename(args.dataset))
                print('Saving model {}'.format(model_name))
                with open(model_name, mode='bw') as modelfile:
                    torch.save(net, modelfile)

def adapt_label_format(y):
    if isinstance(y, list): return adapt_label_format(np.array(y))
    if len(y.shape)==1: return adapt_label_format(y.reshape(-1,1))
    return y

def check_label_format(y):
    assert isinstance(y, np.ndarray), "np.ndarray expected for labels' codification"
    assert len(y.shape)==2, 'shape is not 2'
    if args.classification=='binary':
        assert set(np.unique(y))=={0,1}, 'inconsistent binary label codification'
    elif args.classification=='singlelabel':
        assert np.all(np.sum(y, axis=1)==1), 'inconsistent single-label codification'


def main(args):

    # data = ReviewsDataset.load(args.dataset)
    # data.limit_vocabulary(max_words=args.features)
    data = TextCollectionLoader(dataset='ohsumed', feat_sel=750)

    nF = data.num_features()
    nC = data.num_categories()

    net = Net(input_size=int((nF + 1) * nF / 2), hidden_size=args.hidden, num_classes=nC, normalize=True)
    net = net.cuda() if args.cuda else net

    # Evaluation measures
    mean_absolute_error = nn.L1Loss()

    Xtr, ytr = data.get_devel_set()
    Xte, yte = data.get_test_set()
    if isinstance(Xte, list):
        Xte=Xte[0]
        yte=yte[0]

    ytr = adapt_label_format(ytr)
    yte = adapt_label_format(yte)
    check_label_format(ytr)
    check_label_format(yte)

    train(Xtr, ytr, net, evaluation_measure=mean_absolute_error, Xte=Xte, yte=yte, num_steps=args.iter,
          learning_rate=args.lr, weight_decay=args.weight_decay)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Text Quantification Net')
    parser.add_argument('--iter', type=int, default=10000, metavar='I',
                        help='number of iterations to train (default: 10000)')
    parser.add_argument('--hidden', type=int, default=512, metavar='H',
                        help='hidden size (default: 512)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--interactive', action='store_true', default=False,
                        help='enables interactive parameter setting')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--dataset', '--d', type=str, metavar='D', required=True,
                        help='path to pickled dataset to load')
    parser.add_argument('--features', '--f', type=int, metavar='F', default=1000,
                        help='feature selection to reduce the number of dimensions (default: 1000)')
    parser.add_argument('--classification', '--c', type=str, metavar='C', required=True, choices=['singlelabel', 'multilabel', 'binary'],
                        help='specifies the classification mode, i.e., singlelabel, multilabel, or binary')
    parser.add_argument('--modeldir', '--m', default='../model', type=str, metavar='M',
                        help='directory where to save best model parameters (default: ../model)')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('CUDA:', args.cuda)

    if not exists(args.modeldir):
        makedirs(args.modeldir)

    main(args)




