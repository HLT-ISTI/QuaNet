import numpy as np
import torch
from keras.datasets import imdb
from keras.preprocessing import sequence
from time import time
import torch.cuda

from nets.quantification import LSTMQuantificationNet

max_features = 5000
max_len = 120
classes = 2

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

min_sample_length = 1000
max_sample_length = 1000

use_cuda = True


def choices(values, k):
    return values[np.random.choice(values.shape[0], k, replace=False)]


def variable(tensor):
    var = torch.autograd.Variable(tensor)
    return var.cuda() if use_cuda else var


def create_batch(yhat, y, batch_size=1000, sample_length=1000):
    batch_prevalences = np.random.random(batch_size)

    yhat_pos = yhat[y == 1]
    yhat_neg = yhat[y != 1]

    batch_y = list()
    batch_yhat = list()
    real_prevalences = list()
    for prevalence in batch_prevalences:
        sample_pos_count = int(sample_length * prevalence)
        if sample_pos_count == sample_length:
            sample_pos_count = sample_length - 1
        if sample_pos_count == 0:
            sample_pos_count = 1
        sample_neg_count = sample_length - sample_pos_count
        real_prevalences.append(sample_pos_count / sample_length)

        sample_yhat = np.concatenate((choices(yhat_pos, k=sample_pos_count), choices(yhat_neg, k=sample_neg_count)))
        sample_y = np.concatenate((np.asarray([[1, 0]] * sample_pos_count, dtype=np.float),
                                   np.asarray([[0, 1]] * sample_neg_count, dtype=np.float)))

        order = np.argsort(sample_yhat[:, 0])
        sample_yhat = sample_yhat[order]
        sample_y = sample_y[order]

        # paired = list(zip(sample_yhat, sample_y))
        # paired = sorted(paired, key=lambda x: x[0])
        # sample_yhat, sample_y = zip(*paired)
        batch_yhat.append(sample_yhat)
        batch_y.append(sample_y)

    batch_yhat_var = variable(torch.FloatTensor(batch_yhat).view(-1, sample_length, 2))
    batch_y_var = variable(torch.FloatTensor(batch_y).view(-1, sample_length, 2))
    real_prevalences = np.asarray(real_prevalences)
    batch_p_var = variable(torch.FloatTensor(np.vstack([real_prevalences, 1 - real_prevalences]).transpose()))

    return batch_yhat_var, batch_y_var, batch_p_var

def create_batch_(yhat_pos, yhat_neg, batch_size=1000, sample_length=1000):
    batch_prevalences = np.random.random(batch_size)

    batch_y = list()
    batch_yhat = list()
    real_prevalences = list()
    for prevalence in batch_prevalences:
        sample_pos_count = int(sample_length * prevalence)
        if sample_pos_count == sample_length:
            sample_pos_count = sample_length - 1
        if sample_pos_count == 0:
            sample_pos_count = 1
        sample_neg_count = sample_length - sample_pos_count
        real_prevalences.append(sample_pos_count / sample_length)

        sample_yhat = np.concatenate((choices(yhat_pos, k=sample_pos_count), choices(yhat_neg, k=sample_neg_count)))
        pos_neg_code = np.array([[1, 0], [0, 1]])
        sample_y = np.repeat(pos_neg_code, repeats=[sample_pos_count, sample_neg_count], axis=0)

        order = np.argsort(sample_yhat[:, 0])
        sample_yhat = sample_yhat[order]
        sample_y = sample_y[order]

        batch_yhat.append(sample_yhat)
        batch_y.append(sample_y)

    batch_yhat_var = variable(torch.FloatTensor(batch_yhat).view(-1, sample_length, 2))
    batch_y_var = variable(torch.FloatTensor(batch_y).view(-1, sample_length, 2))
    real_prevalences = np.asarray(real_prevalences)
    batch_p_var = variable(torch.FloatTensor(np.vstack([real_prevalences, 1 - real_prevalences]).transpose()))

    return batch_yhat_var, batch_y_var, batch_p_var


quant_lstm_hidden_size = 32
quant_lstm_layers = 1
quant_lin_layers_sizes = [32, 16]

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


class_steps = 3000
with open('class_net_' + str(class_steps) + '.pt', mode='br') as modelfile:
    class_net = torch.load(modelfile)

if use_cuda:
    class_net = class_net.cuda()
else:
    class_net = class_net.cpu()



class_net.eval()
val_yhat = list()
test_yhat = list()
batch_size = 100
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
test_every = 100
save_every = 1000

test_samples = 100


def get_name(step):
    filename = 'quant_net_' + str(step)
    return filename + '.pt'


quant_net = LSTMQuantificationNet(classes, quant_lstm_hidden_size, quant_lstm_layers, quant_lin_layers_sizes)

if use_cuda:
    quant_net = quant_net.cuda()

print(quant_net)

lr = 0.0001
weight_decay = 0.0001

quant_optimizer = torch.optim.Adam(quant_net.parameters(), lr=lr, weight_decay=weight_decay)

batch_size = 1000
sample_length = 1000

print('init quantification')
with open('quant_net_hist.txt', mode='w', encoding='utf-8') as outputfile, \
        open('quant_net_test.txt', mode='w', encoding='utf-8') as testoutputfile:
    quant_loss_sum = 0
    t_init = time()
    val_yhat_pos = val_yhat[val_y == 1]
    val_yhat_neg = val_yhat[val_y != 1]
    for step in range(1, quant_steps + 1):

        batch_yhat, batch_y, batch_p = create_batch_(val_yhat_pos, val_yhat_neg, batch_size, sample_length)

        quant_optimizer.zero_grad()

        quant_net.train()

        batch_phat = quant_net.forward(batch_yhat)
        quant_loss = quant_loss_function(batch_phat, batch_p)

        quant_loss.backward()

        quant_optimizer.step()

        quant_loss_sum += quant_loss.data[0]

        if step % status_every == 0:
            print('step {}\tloss {:.5}\tv {:.2f}'.format(step,quant_loss_sum / status_every, status_every/(time()-t_init)))
            quant_loss_sum = 0
            t_init = time()

        if step % test_every == 0:
            quant_net.eval()

            test_batch_yhat, test_batch_y, test_batch_p = create_batch(test_yhat, test_y, test_samples,
                                                                       sample_length)
            test_batch_phat = quant_net.forward(test_batch_yhat)

            for i in range(test_samples):
                net_prev = float(test_batch_phat[i, 0])
                cc_prev = classify_and_count(np.asarray(test_batch_yhat[i, :, :].data))
                if val_tpr - val_fpr != 0:
                    acc_prev = (cc_prev - val_fpr) / (val_tpr - val_fpr)
                    anet_prev = (net_prev - val_fpr) / (val_tpr - val_fpr)
                else:
                    acc_prev = -1.
                    anet_prev = -1.
                print('step {} p={} ccp={} accp={} netp={} anetp={}'
                      .format(step, test_batch_p[i,0].data[0], cc_prev, acc_prev, net_prev, anet_prev))

        if step % save_every == 0:
            filename = get_name(step)
            print('saving to', filename)
            with open(filename, mode='bw') as modelfile:
                torch.save(class_net, modelfile)
                torch.save(quant_net, modelfile)
