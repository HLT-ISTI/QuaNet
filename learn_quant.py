from time import time

import numpy as np
import torch
import torch.cuda
import torch.cuda
from keras.datasets import imdb
from keras.preprocessing import sequence

# from inntt import *
from nets.quantification import LSTMQuantificationNet
from plot_correction import plot_corr, plot_loss, plot_bins

interactive = False

max_features = 5000
max_len = 120
classes = 2
MAX_SAMPLE_LENGTH = 500

print('Loading data...')
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=max_features)


def split_train_validation(x, y, val_portion):
    x_pos = x[y == 1]
    x_neg = x[y != 1]
    pos_split = int(len(x_pos) * (1 - val_portion))
    neg_split = int(len(x_neg) * (1 - val_portion))
    train_x = np.concatenate((x_pos[:pos_split], x_neg[:neg_split]))
    train_y = np.asarray([1] * pos_split + [0] * neg_split)
    val_x = np.concatenate((x_pos[pos_split:], x_neg[neg_split:]))
    val_y = np.asarray([1] * (len(x_pos) - pos_split) + [0] * (len(x_neg) - neg_split))
    return train_x, train_y, val_x, val_y


val_portion = 0.4

train_x, train_y, val_x, val_y = split_train_validation(train_x, train_y, val_portion)

print(len(train_x), 'train docs')
print(len(val_x), 'validation docs')
print(len(test_x), 'test docs')

train_x = sequence.pad_sequences(train_x, maxlen=max_len)
val_x = sequence.pad_sequences(val_x, maxlen=max_len)
test_x = sequence.pad_sequences(test_x, maxlen=max_len)
print('x_train shape:', train_x.shape)
print('x_val shape:', val_x.shape)
print('x_test shape:', test_x.shape)

# min_sample_length = 1000
# max_sample_length = 1000

use_cuda = True


def choices(values, k):
    return values[np.random.choice(values.shape[0], k, replace=False)]


def variable(tensor):
    var = torch.autograd.Variable(tensor)
    return var.cuda() if use_cuda else var


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


quant_lstm_hidden_size = 64
quant_lstm_layers = 1
quant_lin_layers_sizes = [1024]  # [128,64,32]

stats_in_lin_layers = True
stats_in_sequence = True

quant_loss_function = torch.nn.MSELoss()


def classify_and_count(yhat):
    return (yhat[:, 0] > 0.5).sum() / len(yhat)


def probabilistic_classify_and_count(yhat):
    return yhat[:, 0].sum() / len(yhat)


def accuracy(yhat, y):
    return ((y + (yhat[:, 0] > 0.5)) % 2 == 2).sum() / len(y)


def tpr(yhat, y):
    return ((y * 2 + (yhat[:, 0] > 0.5)) == 3).sum() / y.sum()


def fpr(yhat, y):
    return ((y * 2 + (yhat[:, 0] > 0.5)) == 1).sum() / (y == 0).sum()


def adjusted_quantification(estim, tpr, fpr, clip=True):
    if (val_tpr - val_fpr) == 0:
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


class_steps = 20000
with open('class_net_' + str(class_steps) + '.pt', mode='br') as modelfile:
    class_net = torch.load(modelfile)

if use_cuda:
    class_net = class_net.cuda()
else:
    class_net = class_net.cpu()

use_document_embeddings_from_classifier = True

class_net.eval()
val_yhat = list()
test_yhat = list()
batch_size = 500
if use_document_embeddings_from_classifier:
    print('creating val_yhat')
    for i in range(0, val_x.shape[0], batch_size):
        yhat, doc_embeddings = class_net.forward(
            variable(torch.LongTensor(val_x[i:i + batch_size]).transpose(0, 1)),
            use_document_embeddings_from_classifier)
        val_yhat.extend(torch.cat((yhat, doc_embeddings), dim=1).data.tolist())
    val_yhat = np.asarray(val_yhat)

    print('creating test_yhat')
    for i in range(0, test_x.shape[0], batch_size):
        yhat, doc_embeddings = class_net.forward(
            variable(torch.LongTensor(test_x[i:i + batch_size]).transpose(0, 1)),
            use_document_embeddings_from_classifier)
        test_yhat.extend(torch.cat((yhat, doc_embeddings), dim=1).data.tolist())
    test_yhat = np.asarray(test_yhat)
else:
    print('creating val_yhat')
    for i in range(0, val_x.shape[0], batch_size):
        val_yhat.extend(
            class_net.forward(
                variable(torch.LongTensor(val_x[i:i + batch_size]).transpose(0, 1))).data.tolist())
    val_yhat = np.asarray(val_yhat)

    print('creating test_yhat')
    for i in range(0, test_x.shape[0], batch_size):
        test_yhat.extend(
            class_net.forward(
                variable(torch.LongTensor(test_x[i:i + batch_size]).transpose(0, 1))).data.tolist())
    test_yhat = np.asarray(test_yhat)

val_tpr = tpr(val_yhat, val_y)
val_fpr = fpr(val_yhat, val_y)

quant_steps = 10000
status_every = 10
test_every = 200
save_every = 1000

test_samples = 1000


def get_name(step):
    filename = 'quant_net_' + str(step)
    return filename + '.pt'


input_size = classes

if use_document_embeddings_from_classifier:
    input_size += 32

quant_net = LSTMQuantificationNet(classes, input_size, quant_lstm_hidden_size, quant_lstm_layers,
                                  quant_lin_layers_sizes, stats_in_lin_layers=stats_in_lin_layers,
                                  stats_in_sequence=stats_in_sequence, drop_p=0.5)

if use_cuda:
    quant_net = quant_net.cuda()

print(quant_net)

lr = 0.0001
weight_decay = 0.00001

quant_optimizer = torch.optim.Adam(quant_net.parameters(), lr=lr, weight_decay=weight_decay)

batch_size = 100

print('init quantification')
with open('quant_net_hist.txt', mode='w', encoding='utf-8') as outputfile, \
        open('quant_net_test.txt', mode='w', encoding='utf-8') as testoutputfile:
    # if interactive:
    #    innt = InteractiveNeuralTrainer()
    #    innt.add_optim_param_adapt('ws', quant_optimizer, 'lr', inc_factor=10.)
    #    innt.add_optim_param_adapt('da', quant_optimizer, 'weight_decay', inc_factor=2.)
    #    innt.start()

    quant_loss_sum = 0
    t_init = time()
    val_yhat_pos = val_yhat[val_y == 1]
    val_yhat_neg = val_yhat[val_y != 1]
    test_yhat_pos = test_yhat[test_y == 1]
    test_yhat_neg = test_yhat[test_y != 1]
    losses = []
    for step in range(1, quant_steps + 1):

        sample_length = min(10 + step // 10, MAX_SAMPLE_LENGTH)

        batch_yhat, batch_y, batch_p, stats = create_batch_(val_yhat_pos, val_yhat_neg, val_tpr, val_fpr, input_size,
                                                            batch_size, sample_length)

        quant_optimizer.zero_grad()

        quant_net.train()

        batch_phat = quant_net.forward(batch_yhat, stats)
        quant_loss = quant_loss_function(batch_phat, batch_p)

        quant_loss.backward()

        quant_optimizer.step()

        loss = quant_loss.data[0]
        quant_loss_sum += loss
        losses.append(loss)

        if step % status_every == 0:
            print('step {}\tloss {:.5}\tv {:.2f}'.format(step, quant_loss_sum / status_every,
                                                         status_every / (time() - t_init)))
            quant_loss_sum = 0
            t_init = time()

        if step % test_every == 0:
            quant_net.eval()

            test_batch_yhat, test_batch_y, test_batch_p, stats = create_batch_(test_yhat_pos, test_yhat_neg, val_tpr,
                                                                               val_fpr, input_size, test_samples,
                                                                               sample_length)
            test_batch_phat = quant_net.forward(test_batch_yhat, stats)

            prevs, net_prevs, cc_prevs, pcc_prevs, acc_prevs, apcc_prevs, anet_prevs = [], [], [], [], [], [], []
            for i in range(test_samples):
                net_prev = float(test_batch_phat[i, 0])
                cc_prev = classify_and_count(np.asarray(test_batch_yhat[i, :, :].data))
                pcc_prev = probabilistic_classify_and_count(np.asarray(test_batch_yhat[i, :, :].data))

                anet_prev = adjusted_quantification(net_prev, val_tpr, val_fpr)
                acc_prev = adjusted_quantification(cc_prev, val_tpr, val_fpr)
                apcc_prev = adjusted_quantification(pcc_prev, val_tpr, val_fpr)

                prevs.append(test_batch_p[i, 0].data[0])

                net_prevs.append(net_prev)
                cc_prevs.append(cc_prev)
                pcc_prevs.append(pcc_prev)
                anet_prevs.append(anet_prev)
                acc_prevs.append(acc_prev)
                apcc_prevs.append(apcc_prev)

                print('step {}\tp={:.3f}\tcc={:.3f}\tacc={:.3f}\tpcc={:.3f}\tapcc={:.3f}\tnet={:.3f}\tanet={:.3f}'
                      .format(step, test_batch_p[i, 0].data[0], cc_prev, acc_prev, pcc_prev, apcc_prev, net_prev,
                              anet_prev))

            print('Average MAE:\tcc={:.5f}\tacc={:.5f}\tpcc={:.5f}\tapcc={:.5f}\tnet={:.5f}\tanet={:.5f}'
                  .format(mae(prevs, cc_prevs), mae(prevs, acc_prevs), mae(prevs, pcc_prevs), mae(prevs, apcc_prevs),
                          mae(prevs, net_prevs), mae(prevs, anet_prevs)))
            print('Average MSE:\tcc={:.5f}\tacc={:.5f}\tpcc={:.5f}\tapcc={:.5f}\tnet={:.5f}\tanet={:.5f}'
                  .format(mse(prevs, cc_prevs), mse(prevs, acc_prevs), mse(prevs, pcc_prevs), mse(prevs, apcc_prevs),
                          mse(prevs, net_prevs), mse(prevs, anet_prevs)))

            # plots
            methods = [cc_prevs, acc_prevs, pcc_prevs, apcc_prevs, net_prevs]
            labels = ['cc', 'acc', 'pcc', 'apcc', 'net']
            plot_corr(prevs, methods, labels, savedir='../plots', savename='corr.png')  # 'step_' + str(step) + '.png')
            plot_bins(prevs, [acc_prevs, apcc_prevs, net_prevs], ['acc', 'apcc', 'net'], mae, bins=10,
                      savedir='../plots', savename='bins_' + mae.__name__ + '.png')
            plot_bins(prevs, [acc_prevs, apcc_prevs, net_prevs], ['acc', 'apcc', 'net'], mse, bins=10,
                      savedir='../plots', savename='bins_' + mse.__name__ + '.png')
            plot_loss(range(step), losses, savedir='../plots', savename='loss.png')

            print('full test')
            tested = 0
            p_ave = []
            true_prev = np.mean(test_y)
            cc = classify_and_count(test_yhat)
            acc = adjusted_quantification(cc, val_tpr, val_fpr)
            pcc = probabilistic_classify_and_count(test_yhat)
            apcc = adjusted_quantification(pcc, val_tpr, val_fpr)
            for test_batch_yhat, test_batch_y, test_batch_p, stats in create_fulltest_batch(test_yhat, test_y, val_tpr,
                                                                                            val_fpr, input_size,
                                                                                            batch_size=batch_size,
                                                                                            sample_length=sample_length):
                test_batch_phat = quant_net.forward(test_batch_yhat, stats)
                p_ave.append(np.mean(test_batch_phat[:, 0].data.cpu().numpy()))
                tested += (test_batch_yhat.shape[0] * test_batch_yhat.shape[1])
                print('\rcomplete {}/{}, cc={:.4f} acc={:.4f} pcc={:.4f} apcc={:.4f} net={:.4f} true_prev={:.4f}'
                      .format(tested, test_yhat.shape[0], cc, acc, pcc, apcc, np.mean(p_ave), true_prev), end='')
            print()

        if step % save_every == 0:
            filename = get_name(step)
            print('saving to', filename)
            with open(filename, mode='bw') as modelfile:
                torch.save(class_net, modelfile)
                torch.save(quant_net, modelfile)
