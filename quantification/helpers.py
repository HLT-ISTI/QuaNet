import numpy as np
import torch
from keras.preprocessing import sequence
from data.rewiews_builder import ReviewsDataset

use_cuda = True
MAX_SAMPLE_LENGTH = 500

class QuantificationResults:
    def __init__(self, train_prev, test_prev):
        self.train_prev=train_prev
        self.test_prev = test_prev
        self.sampletest = {}
        self.fulltest = {}
        self.metrics = set()

    def add_results(self, metric_name, test_name, cc, pcc, acc, apcc, net):
        assert test_name in ["sample", "full"], 'unexpected test_name'
        results_container = self.sampletest if test_name == "sample" else self.fulltest
        results_container[metric_name] = {'cc':cc, 'pcc':pcc, 'acc':acc, 'apcc':apcc, 'net':net}
        self.metrics.add(metric_name)

    def get(self, metric_name, test_name, method_name):
        results_container = self.sampletest if test_name == "sample" else self.fulltest
        return results_container[metric_name][method_name]

    def header(self):
        strbuilder = []
        metrics = sorted(list(self.metrics))
        for metric in metrics:
            for mode in ["sample", "full"]:
                for method in ['cc','pcc','acc','apcc','net']:
                    strbuilder.append('-'.join([metric,mode,method]))
        strbuilder.append('train_prev')
        strbuilder.append('test_prev')
        return '\t'.join(strbuilder)

    def show(self):
        strbuilder = []
        metrics = sorted(list(self.metrics))
        for metric in metrics:
            for mode in ["sample", "full"]:
                for method in ['cc','pcc','acc','apcc','net']:
                    strbuilder.append(self.get(metric,mode,method))
        strbuilder.append(self.train_prev)
        strbuilder.append(self.test_prev)
        return '\t'.join(['%.5f'%x for x in strbuilder])




def variable(tensor):
    var = torch.autograd.Variable(tensor)
    return var.cuda() if use_cuda else var

def loadDataset(dataset, vocabularysize=5000, val_portion = 0.4, max_len = 120):
    print('Loading dataset '+dataset)
    if dataset == 'imdb':
        from keras.datasets import imdb
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocabularysize)
    else:
        data = ReviewsDataset.load(dataset)
        data.limit_vocabulary(vocabularysize)
        (x_train, y_train), (x_test, y_test) = (np.array(data.Xtr), data.ytr), (np.array(data.Xte), data.yte)

    x_train, y_train, x_val, y_val = split_train_validation(x_train, y_train, val_portion)

    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_val = sequence.pad_sequences(x_val, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def choices(values, k):
    replace = True if k>values.shape[0] else False # cannot take more elements than the existing ones if replace=False
    return values[np.random.choice(values.shape[0], k, replace=replace)]

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

def f1(y_hard_true, y_soft_pred):
    y_hard_true = todata(y_hard_true)
    y_soft_pred = todata(y_soft_pred)
    pred = (y_soft_pred[:, 0] > 0.5)
    truth = (y_hard_true[:, 0] > 0.5)
    tp = torch.sum(pred[truth == 1])
    fp = torch.sum(pred[truth != 1])
    fn = torch.sum(truth[pred != 1])
    f1 = 2*tp/(2*tp+fp+fn)
    return f1


def classify_and_count(yhat):
    return (yhat[:, 0] > 0.5).sum() / len(yhat)

def probabilistic_classify_and_count(yhat):
    return yhat[:, 0].sum() / len(yhat)

def tpr(yhat, y):
    return ((y * 2 + (yhat[:, 0] > 0.5)) == 3).sum() / y.sum()


def fpr(yhat, y):
    return ((y * 2 + (yhat[:, 0] > 0.5)) == 1).sum() / (y == 0).sum()


def adjusted_quantification(estim, tpr, fpr, clip=True):
    if (tpr - fpr) == 0:
        return -1
    adjusted = (estim - fpr) / (tpr - fpr)
    if clip:
        adjusted = max(min(adjusted, 1.), 0.)
    return adjusted


def mae(prevs, method):
    assert len(prevs) == len(method), 'wrong sizes'
    diff = np.array([prevs[i] - method[i] for i in range(len(prevs))])
    return np.mean(np.abs(diff))


def mse(prevs, method):
    assert len(prevs) == len(method), 'wrong sizes'
    diff = np.array([prevs[i] - method[i] for i in range(len(prevs))])
    return np.mean(diff ** 2.)

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
    return x[y==1], x[y!=1]

def sample_data(x_pos, x_neg, prevalence, batch_size):
    pos_count = int(batch_size * prevalence)
    neg_count = batch_size - pos_count
    prevalence = pos_count / batch_size

    sampled_pos = x_pos[np.random.choice(x_pos.shape[0], pos_count)]
    sampled_neg = x_neg[np.random.choice(x_neg.shape[0], neg_count)]

    sampled_x = np.vstack((sampled_pos, sampled_neg))
    sampled_y = np.array([1]*pos_count+[0]*neg_count)

    order = np.random.permutation(pos_count + neg_count)
    sampled_x = sampled_x[order]
    sampled_y = sampled_y[order]

    return sampled_x, sampled_y, prevalence

def prepare_classification(x, y):
    y = np.vstack((y,1-y)).T
    xvar = variable(torch.LongTensor(x).transpose(0, 1))
    yvar = variable(torch.FloatTensor(y))
    return xvar, yvar


def quantification_batch(yhat_pos, yhat_neg, val_tpr, val_fpr, input_size, batch_size=1000, sample_length=1000):
    # batch_prevalences = np.random.random(batch_size)*0.8+0.1
    batch_prevalences = np.random.random(batch_size)

    batch_y = list()
    batch_yhat = list()
    real_prevalences = list()
    stats = list()
    for prevalence in batch_prevalences:
        sample_pos_count = int(sample_length * prevalence)
        sample_neg_count = sample_length - sample_pos_count
        real_prevalences.append(sample_pos_count / sample_length)

        if sample_pos_count == sample_length:
            sample_yhat = choices(yhat_pos, k=sample_pos_count)
        elif sample_pos_count == 0:
            sample_yhat = choices(yhat_neg, k=sample_neg_count)
        else:
            sample_yhat = np.concatenate((choices(yhat_pos, k=sample_pos_count), choices(yhat_neg, k=sample_neg_count)))
        pos_neg_code = np.array([[1., 0.], [0., 1.]])
        sample_y = np.repeat(pos_neg_code, repeats=[sample_pos_count, sample_neg_count], axis=0)

        order = np.argsort(sample_yhat[:, 0])
        sample_yhat = sample_yhat[order]
        sample_y = sample_y[order]

        cc = classify_and_count(sample_yhat)
        acc = adjusted_quantification(cc, val_tpr, val_fpr, clip=False)
        pcc = probabilistic_classify_and_count(sample_yhat)
        apcc = adjusted_quantification(pcc, val_tpr, val_fpr, clip=False)

        batch_yhat.append(sample_yhat)
        batch_y.append(sample_y)
        stats.append([[cc, 1 - cc], [acc, 1 - acc], [pcc, 1 - pcc], [apcc, 1 - apcc], [val_tpr, 1 - val_tpr],
                      [val_fpr, 1 - val_fpr]])

    stats_var = variable(torch.FloatTensor(stats).view(-1, 6, 2))

    batch_yhat_var = variable(torch.FloatTensor(batch_yhat).view(-1, sample_length, input_size))
    batch_y_var = variable(torch.FloatTensor(batch_y).view(-1, sample_length, 2))
    real_prevalences = np.asarray(real_prevalences)
    batch_p_var = variable(torch.FloatTensor(np.vstack([real_prevalences, 1 - real_prevalences]).transpose()))

    return batch_yhat_var, batch_y_var, batch_p_var, stats_var


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
