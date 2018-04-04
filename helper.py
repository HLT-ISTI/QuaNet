import numpy as np
import torch
from time import time
import os
from keras.preprocessing import sequence
from data.rewiews_builder import ReviewsDataset

use_cuda = True
MAX_SAMPLE_LENGTH = 500

def variable(tensor):
    var = torch.autograd.Variable(tensor)
    return var.cuda() if use_cuda else var

def loadDataset(dataset, max_features=5000, val_portion = 0.4, max_len = 120):
    print('Loading dataset '+dataset)
    if dataset == 'imdb':
        from keras.datasets import imdb
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    elif dataset == 'hp':
        datasets_dir = os.path.join('../datasets/build/online',dataset)
        hp = os.path.join(datasets_dir, 'Seq2004_1OnlineS3F.pkl')
        data = ReviewsDataset.load(hp)
        data.limit_vocabulary(max_features)
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

def accuracy(y_hard_true, y_soft_pred):
    pred = y_soft_pred[:, 0] > 0.5
    truth = y_hard_true[:, 0] > 0.5
    return torch.mean((pred == truth).type(torch.FloatTensor)).data[0]

def _accuracy(yhat, y):
    return ((y + (yhat[:, 0] > 0.5)) % 2 == 2).sum() / len(y)

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

def printtee(msg, fout):
    print(msg)
    fout.write(msg + '\n')
    fout.flush()

def sample_data(x_pos, x_neg, prevalence, batch_size):
    sample_pos_count = int(batch_size * prevalence)
    sample_neg_count = batch_size - sample_pos_count
    prevalence = sample_pos_count / batch_size

    sampled_pos = x_pos[np.random.choice(x_pos.shape[0], sample_pos_count)]
    sampled_neg = x_neg[np.random.choice(x_neg.shape[0], sample_neg_count)]
    sampled_x = np.vstack((sampled_pos,sampled_neg))

    pos_neg_code = np.array([[1,0],[0,1]])
    sampled_y = np.repeat(pos_neg_code, repeats=[sample_pos_count,sample_neg_count], axis=0)

    order = np.random.permutation(sample_pos_count+sample_neg_count)
    sampled_x = sampled_x[order]
    sampled_y = sampled_y[order]

    sampled_x = variable(torch.LongTensor(sampled_x).transpose(0, 1))
    sampled_y = variable(torch.FloatTensor(sampled_y).view(-1, 2))
    prevalence_var = variable(torch.FloatTensor([prevalence, 1 - prevalence]).view([1, 2]))

    return sampled_x, sampled_y, prevalence_var

def create_batch_(yhat_pos, yhat_neg, val_tpr, val_fpr, input_size, batch_size=1000, sample_length=1000):
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

def predict(class_net, x, use_document_embeddings_from_classifier):
    #mode=class_net.training
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

    #class_net.train(mode)
    return np.asarray(yhat)

#
# def predict(class_net, x, use_document_embeddings_from_classifier):
#     class_net.eval()
#     val_yhat = list()
#     test_yhat = list()
#     batch_size = 500
#     if use_document_embeddings_from_classifier:
#         print('creating val_yhat')
#         for i in range(0, x_val.shape[0], batch_size):
#             yhat, doc_embeddings = class_net.forward(
#                 variable(torch.LongTensor(x_val[i:i + batch_size]).transpose(0, 1)),
#                 use_document_embeddings_from_classifier)
#             val_yhat.extend(torch.cat((yhat, doc_embeddings), dim=1).data.tolist())
#         val_yhat = np.asarray(val_yhat)
#
#         print('creating test_yhat')
#         for i in range(0, x_test.shape[0], batch_size):
#             yhat, doc_embeddings = class_net.forward(
#                 variable(torch.LongTensor(x_test[i:i + batch_size]).transpose(0, 1)),
#                 use_document_embeddings_from_classifier)
#             test_yhat.extend(torch.cat((yhat, doc_embeddings), dim=1).data.tolist())
#         test_yhat = np.asarray(test_yhat)
#     else:
#         print('creating val_yhat')
#         for i in range(0, x_val.shape[0], batch_size):
#             val_yhat.extend(
#                 class_net.forward(
#                     variable(torch.LongTensor(x_val[i:i + batch_size]).transpose(0, 1))).data.tolist())
#         val_yhat = np.asarray(val_yhat)
#
#         print('creating test_yhat')
#         for i in range(0, x_test.shape[0], batch_size):
#             test_yhat.extend(
#                 class_net.forward(
#                     variable(torch.LongTensor(x_test[i:i + batch_size]).transpose(0, 1))).data.tolist())
#         test_yhat = np.asarray(test_yhat)
#
#     return val_yhat, test_yhat
