import numpy as np
import torch
import itertools
import os
from sklearn.externals.joblib import Parallel, delayed
from keras.preprocessing import sequence
from scipy.sparse import issparse, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from data.rewiews_builder import ReviewsDataset
from quantification.constants import *


def variable(tensor):
    var = torch.autograd.Variable(tensor)
    return var.cuda() if use_cuda else var


def loadDataset(dataset, vocabularysize=5000, val_portion = 0.4, max_len = 120, flatten_test=True, mode='sequence',
                datapath='../datasets/build/single'):
    """
    Loads a sentiment dataset (imdb, hp, or kindle) and process it. Documents are represented as sequences of word-ids.
    For hp and kindle, the test set is a list of test-sets corresponding to different years, unless flatten_test is True.
    For hp and kindle, the 'single' version at split point 3 (filtered) is considered. For hp the training slots
    are 2, and 3 for kindle.
    More information about the hp and kindle datasets can be found in the dataset builders.
    :param dataset: imdb, hp, or kindle
    :param vocabularysize: the size of the vocabulary; all other words will be replaced by the id of the UNK token
    :param val_portion: validation proportion of the development set
    :param max_len: maximum length of words for documents, which will be padded to this value
    :param flatten_test: if True, all test sets are flatten into a single set (ignored for imdb)
    :param mode: selects 'sequence' for sequence of ids or 'matrix' for a csr_matrix with tfidf
    :return: train/validation/test splits, each as a tuple of input/output of the form (x,y); in case the flatten_test
    is set to Truen, the x and y components are lists of inputs and outputs
    """
    assert dataset in ['imdb','hp','kindle'], 'unknown dataset, valid ones are imdb, hp, and kindle'
    print('Loading dataset '+dataset)
    if dataset == 'imdb':
        from keras.datasets import imdb
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocabularysize)
    else:
        tr_slots = '3'
        data = ReviewsDataset.load(os.path.join(datapath,dataset,'SeqSingle'+tr_slots+'S3F.pkl'))
        if mode=='sequence':
            data.limit_vocabulary(vocabularysize)
        (x_train, y_train) = (np.array(data.Xtr), data.ytr)
        if flatten_test:
            x_test = np.array(list(itertools.chain.from_iterable(data.Xte)))
            y_test = np.concatenate(data.yte)
        else:
            x_test = [np.array(Xte_i) for Xte_i in data.Xte]
            y_test = data.yte

    if mode == 'matrix':
        #todo: for kindle and hp we do already have the text versions... load them (though it's equivalent...)
        def from_id2str(sequences):
            return np.array([' '.join([str(x) for x in seq_ids]) for seq_ids in sequences])
        x_train = from_id2str(x_train)
        x_test = from_id2str(x_test)
        tfidf_vect = TfidfVectorizer(min_df=5, sublinear_tf=True)  # stop_words='english' ?
        tfidf_vect.fit(x_train)

    x_train, y_train, x_val, y_val = split_train_validation(x_train, y_train, val_portion)

    if mode == 'sequence':
        x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        x_val = sequence.pad_sequences(x_val, maxlen=max_len)
        if flatten_test:
            x_test = sequence.pad_sequences(x_test, maxlen=max_len)
        else:
            x_test = [sequence.pad_sequences(Xte_i, maxlen=max_len) for Xte_i in x_test]
    else:
        x_train = tfidf_vect.transform(x_train)
        x_val = tfidf_vect.transform(x_val)
        x_test = tfidf_vect.transform(x_test)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def choices(values, k):
    if k == 0:
        return np.empty(0), np.empty(0)
    replace = True if k>values.shape[0] else False # cannot take more elements than the existing ones if replace=False
    indexes = np.random.choice(values.shape[0], k, replace=replace)
    return values[indexes], indexes

def get_name(step, info=''):
    filename = info + 'net_' + str(step)
    return filename + '.pt'

def class_batched_predictions(class_net, x, batchsize=100):
    size = x.shape[1]
    nbatches = size // batchsize
    if size  % batchsize > 0:
        nbatches += 1
    y_accum = []
    for b in range(nbatches):
        y_accum.append(class_net.forward(x[:, b * batchsize:(b + 1) * batchsize]).data)
    return torch.cat(y_accum)

def quant_batched_predictions(quant_net, x, stats, batchsize=100):
    size = x.shape[0]
    nbatches = size // batchsize
    if size  % batchsize > 0:
        nbatches += 1
    y_accum = []
    for b in range(nbatches):
        xbatch = x[b * batchsize:(b + 1) * batchsize]
        statsbatch = stats[b * batchsize:(b + 1) * batchsize] if stats is not None else None
        y_accum.append(quant_net.forward(xbatch, statsbatch).data)
    return torch.cat(y_accum)


def todata(v):
    return v.data if isinstance(v, torch.autograd.Variable) else v

def accuracy(y_hard_true, y_soft_pred):
    y_hard_true = todata(y_hard_true)
    y_soft_pred= todata(y_soft_pred)
    pred = y_soft_pred[:, 0] > 0.5
    truth = y_hard_true[:, 0] > 0.5
    return torch.mean((pred == truth).type(torch.FloatTensor))


def classify_and_count(yhat):
    return (yhat[:,0] > 0.5).sum() / len(yhat)

def probabilistic_classify_and_count(yhat):
    return yhat[:,0].sum() / len(yhat)

def tpr(yhat, y):
    return ((y * 2 + (yhat[:,0] > 0.5)) == 3).sum() / y.sum()

def fpr(yhat, y):
    return ((y * 2 + (yhat[:,0] > 0.5)) == 1).sum() / (y == 0).sum()

def ptpr(yhat, y):
    positives = y.sum()
    ptp = yhat[y==1,0].sum()
    return ptp / positives

def pfpr(yhat, y):
    negatives = (1-y).sum()
    pfp = yhat[y==0,0].sum()
    return pfp / negatives


def adjusted_quantification(estim, tpr, fpr, clip=True):
    if (tpr - fpr) == 0:
        return -1
    adjusted = (estim - fpr) / (tpr - fpr)
    if clip:
        adjusted = max(min(adjusted, 1.), 0.)
    return adjusted

def mae(prevs, prevs_hat):
    return Mean(AE, prevs, prevs_hat)

def mse(prevs, prevs_hat):
    return Mean(SE, prevs, prevs_hat)

def mkld(prevs, prevs_hat):
    return Mean(KLD, prevs, prevs_hat)

def mnkld(prevs, prevs_hat):
    return Mean(NKLD, prevs, prevs_hat)

def mrae(prevs, prevs_hat):
    return Mean(RAE, prevs, prevs_hat)

def Mean(error_metric, prevs, prevs_hat):
    n = len(prevs)
    assert n == len(prevs_hat), 'wrong sizes'
    return np.mean([error_metric(prevs[i], prevs_hat[i]) for i in range(n)])

def AE(p, p_hat):
    return abs(p_hat-p)

def SE(p, p_hat):
    return (p_hat-p)**2

def KLD(p, p_hat, eps=1e-8):
    sp = p+eps
    sp_hat = p_hat + eps
    first = sp*np.log(sp/sp_hat)
    second = (1.-sp)*np.log(abs((1.-sp)/(1.-sp_hat)))
    return first + second

def NKLD(p, p_hat):
    ekld = np.exp(KLD(p, p_hat))
    return 2.*ekld/(1+ekld) - 1.

def RAE(p, p_hat, eps=1/(2*MAX_SAMPLE_LENGTH)): # it was proposed in literature an eps = 1/(2*T), with T the size of the test set
    return abs(p_hat-p+eps)/(p+eps)

def split_train_validation(x, y, val_portion, shuffle=True):
    np.random.seed(23)
    if shuffle:
        order = np.random.permutation(x.shape[0])
        x, y = x[order], y[order]
    x_pos = x[y == 1]
    x_neg = x[y != 1]
    pos_split = int(len(x_pos) * (1 - val_portion))
    neg_split = int(len(x_neg) * (1 - val_portion))
    x_train = np.concatenate((x_pos[:pos_split], x_neg[:neg_split]))
    y_train = np.asarray([1] * pos_split + [0] * neg_split)
    x_val = np.concatenate((x_pos[pos_split:], x_neg[neg_split:]))
    y_val = np.asarray([1] * (len(x_pos) - pos_split) + [0] * (len(x_neg) - neg_split))
    return x_train, y_train, x_val, y_val

def split_pos_neg(x,y):
    ids = np.arange(x.shape[0])
    return x[y==1], x[y!=1], ids[y==1], ids[y!=1]


def sample_data(x_pos, x_neg, prevalence, sample_size):
    sparse = issparse(x_pos)
    pos_count = int(sample_size * prevalence)
    neg_count = sample_size - pos_count
    real_prevalence = pos_count / sample_size

    sampled_pos = x_pos[np.random.choice(x_pos.shape[0], pos_count)]
    sampled_neg = x_neg[np.random.choice(x_neg.shape[0], neg_count)]

    if sparse:
        sampled_x = vstack((sampled_pos, sampled_neg))
    else:
        sampled_x = np.vstack((sampled_pos, sampled_neg))
    sampled_y = np.array([1]*pos_count+[0]*neg_count)

    order = np.random.permutation(pos_count + neg_count)
    sampled_x = sampled_x[order]
    sampled_y = sampled_y[order]

    if sparse:
        sampled_x.sort_indices()

    return sampled_x, sampled_y, real_prevalence

def prepare_classification(x, y):
    y = np.vstack((y,1-y)).T
    xvar = variable(torch.LongTensor(x).transpose(0, 1))
    yvar = variable(torch.FloatTensor(y))
    return xvar, yvar

def define_prev_range(with_limits):
    if with_limits:
        prevs_range = np.arange(21) * 1 / 20
        prevs_range[0] += 0.01
        prevs_range[-1] -= 0.01
    else:
        prevs_range = 1 / 20 + np.arange(19) * 1 / 20  # [0.05, 0.1, 0.15, ... , 0.95]
    return prevs_range

def quantification_uniform_sampling(ids_pos, ids_neg, yhat_pos, yhat_neg, val_tpr, val_fpr, val_ptpr, val_pfpr, input_size,
                                    n_samples, sample_size, prevs_range, seed=None):
    if seed is not None:
        print('setting seed {}'.format(seed))
        np.random.seed(seed)

    batch_prevalences = np.repeat(prevs_range, n_samples / prevs_range.size)

    batch_y = list()
    batch_yhat = list()
    real_prevalences = list()
    stats = list()
    ids_chosen = []
    for prevalence in batch_prevalences:
        sample_pos_count = int(sample_size * prevalence)
        sample_neg_count = sample_size - sample_pos_count
        real_prevalences.append(sample_pos_count / sample_size)

        ids_pos_chosen, pos_indexes = choices(ids_pos, k=sample_pos_count)
        ids_neg_chosen, neg_indexes = choices(ids_neg, k=sample_neg_count)
        if sample_pos_count==0:
            sample_yhat = yhat_neg[neg_indexes]
        elif sample_neg_count == 0:
            sample_yhat = yhat_pos[pos_indexes]
        else:
            sample_yhat = np.concatenate((yhat_pos[pos_indexes], yhat_neg[neg_indexes]))
        ids_chosen.append(np.concatenate((ids_pos_chosen, ids_neg_chosen)))

        pos_neg_code = np.array([[1., 0.], [0., 1.]])
        #pos_neg_code = np.array([1., 0.])
        sample_y = np.repeat(pos_neg_code, repeats=[sample_pos_count, sample_neg_count], axis=0)

        order = np.argsort(sample_yhat[:,0])
        sample_yhat = sample_yhat[order]
        sample_y = sample_y[order]

        cc = classify_and_count(sample_yhat)
        acc = adjusted_quantification(cc, val_tpr, val_fpr, clip=False)
        pcc = probabilistic_classify_and_count(sample_yhat)
        apcc = adjusted_quantification(pcc, val_ptpr, val_pfpr, clip=False)

        batch_yhat.append(sample_yhat)
        batch_y.append(sample_y)
        stats.append([[cc, 1 - cc], [acc, 1 - acc], [pcc, 1 - pcc], [apcc, 1 - apcc], [val_tpr, 1 - val_tpr],
                      [val_fpr, 1 - val_fpr], [val_ptpr, 1 - val_ptpr], [val_pfpr, 1 - val_pfpr]])

    stats_var = variable(torch.FloatTensor(stats).view(-1, 8, 2))

    batch_yhat_var = variable(torch.FloatTensor(batch_yhat).view(-1, sample_size, input_size))
    batch_y_var = variable(torch.FloatTensor(batch_y).view(-1, sample_size, 2))
    #batch_y_var = variable(torch.FloatTensor(batch_y).view(-1, sample_size, 1))
    real_prevalences = np.asarray(real_prevalences)
    batch_p_var = variable(torch.FloatTensor(np.vstack([real_prevalences, 1 - real_prevalences]).transpose()))
    #batch_p_var = variable(torch.FloatTensor(real_prevalences))

    ids_chosen = np.vstack(ids_chosen).astype(int)
    return batch_yhat_var, batch_y_var, batch_p_var, stats_var, ids_chosen

@DeprecationWarning
def create_fulltest_batch(yhat, y, val_tpr, val_fpr, input_size, batch_size=1000, sample_length=MAX_SAMPLE_LENGTH):
    ntest = yhat.shape[0]

    permutation = np.random.permutation(ntest)
    nsplits = ntest // sample_length
    # if ntest % sample_length > 0:
    #     nsplits+=1

    batch_y, batch_yhat, real_prevalences, stats = [], [], [], []
    batched = 0
    for split in range(nsplits):
        indices = permutation[split * sample_length:(split + 1) * sample_length]

        sample_yhat = yhat[indices]
        sample_y = y[indices]

        order = np.argsort(sample_yhat[:, 0])
        sample_yhat = sample_yhat[order]
        sample_y = sample_y[order]

        cc = classify_and_count(sample_yhat)
        acc = adjusted_quantification(cc, val_tpr, val_fpr, clip=False)
        pcc = probabilistic_classify_and_count(sample_yhat)
        apcc = adjusted_quantification(pcc, val_tpr, val_fpr, clip=False)

        batch_yhat.append(sample_yhat)
        batch_y.append(np.vstack((sample_y, 1. - sample_y)).T)
        stats.append([[cc, 1 - cc], [acc, 1 - acc], [pcc, 1 - pcc], [apcc, 1 - apcc], [val_tpr, 1 - val_tpr],
                      [val_fpr, 1 - val_fpr]])
        real_prevalences.append(np.mean(sample_y))

        batched += 1
        if batched == batch_size or split == nsplits - 1:
            stats_var = variable(torch.FloatTensor(stats).view(-1, 6, 2))
            batch_yhat_var = variable(torch.FloatTensor(batch_yhat).view(-1, sample_length, input_size))
            batch_y_var = variable(torch.FloatTensor(batch_y).view(-1, sample_length, 2))
            real_prevalences = np.asarray(real_prevalences)
            batch_p_var = variable(torch.FloatTensor(np.vstack([real_prevalences, 1 - real_prevalences]).transpose()))

            yield batch_yhat_var, batch_y_var, batch_p_var, stats_var

            batch_y, batch_yhat, real_prevalences, stats = [], [], [], []
            batched = 0

    raise StopIteration()

def predict(class_net, x, use_document_embeddings_from_classifier=False):
    mode=class_net.training
    class_net.eval()
    yhat = list()
    batch_size = 500

    for i in range(0, x.shape[0], batch_size):
        if use_document_embeddings_from_classifier:
            yhat_, doc_embeddings = class_net.forward(
                variable(torch.LongTensor(x[i:i + batch_size]).transpose(0, 1)),
                use_document_embeddings_from_classifier)
            yhat.extend(torch.cat((yhat_, doc_embeddings), dim=1).data.tolist())
        else:
            yhat.extend(
                class_net.forward(
                    variable(torch.LongTensor(x[i:i + batch_size]).transpose(0, 1))).data.tolist())

    class_net.train(mode)
    return np.asarray(yhat)

def adjust_learning_rate(optimizer, iter, each, initial_lr):
    # sets the learning rate to the initial LR decayed by 0.1 every 'each' iterations
    lr = initial_lr * (0.1 ** (iter // each))
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)
    return lr



def compute_true_prevalence(test_batch_p):
    prevs = test_batch_p[:, 0].data
    #prevs = test_batch_p.data
    return prevs.cpu().numpy() if use_cuda else prevs.numpy()

def compute_classify_count(test_batch_yhat, val_tpr, val_fpr, probabilistic):

    test_samples = test_batch_yhat.shape[0]
    quantifier = probabilistic_classify_and_count if probabilistic else classify_and_count

    prevs_hat = []
    for i in range(test_samples):
        prevs_hat.append(quantifier(np.asarray(test_batch_yhat[i, :, :].data)))

    adjusted_prevs_hat = [adjusted_quantification(prev_hat, val_tpr, val_fpr) for prev_hat in prevs_hat]

    return prevs_hat, adjusted_prevs_hat


def compute_baseline(Q, data_matrix, test_ids_choices, prev_range, optim_params, error=mse, n_jobs=-1):
    print('\tComputing baseline-'+Q.__name__)

    (Xtr, ytr), (Xva, yva), (Xte, yte) = data_matrix
    print('\toptimizing in param space {}'.format(optim_params))
    Q.optim(Xtr, ytr, Xva, yva, error,
            n_samples=100,
            sample_size=MAX_SAMPLE_LENGTH,
            prev_range=prev_range,
            parameters=optim_params,
            refit=True)

    p_estim = Parallel(verbose=1, n_jobs=n_jobs)\
        (delayed(Q.predict)(Xte[sample_ids],yte[sample_ids]) for sample_ids in test_ids_choices)

    return p_estim

