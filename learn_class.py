import random
import numpy as np
import torch
from keras.datasets import imdb
from keras.preprocessing import sequence
from time import time
import os
from data.rewiews_builder import ReviewsDataset
from nets.classification import LSTMTextClassificationNet

max_features = 5000
max_len = 120
classes = 2
MAX_SAMPLE_LENGTH = 500

dataset = 'hp'

print('Loading dataset '+dataset)
if dataset == 'imdb':
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
elif dataset == 'hp':
    datasets_dir = os.path.join('../datasets/build/online',dataset)
    hp = os.path.join(datasets_dir, 'Seq2000_1OnlineS3F.pkl')
    data = ReviewsDataset.load(hp)
    (x_train, y_train), (x_test, y_test) = (np.array(data.Xtr), data.ytr), (np.array(data.Xte), data.yte)


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


val_portion = 0.4

x_train, y_train, x_val, y_val = split_train_validation(x_train, y_train, val_portion)

print(len(x_train), 'train docs')
print(len(x_val), 'validation docs')
print(len(x_test), 'test docs')

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_val = sequence.pad_sequences(x_val, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape:', x_test.shape)

use_cuda = True


def choices(list, k):
    return np.random.permutation(list)[:k]


def variable(tensor):
    var = torch.autograd.Variable(tensor)
    return var.cuda() if use_cuda else var


def sample_data(x, y, prevalence, batch_size):
    x_pos = x[y == 1]
    x_neg = x[y != 1]
    sample_pos_count = int(batch_size * prevalence)
    sample_neg_count = batch_size - sample_pos_count
    prevalence = sample_pos_count / batch_size

    sampled_x = list()
    sampled_x.extend(choices(x_pos, k=sample_pos_count).tolist())
    sampled_x.extend(choices(x_neg, k=sample_neg_count).tolist())

    sampled_y = list()
    sampled_y.extend([[1, 0]] * sample_pos_count)
    sampled_y.extend([[0, 1]] * sample_neg_count)

    paired = list(zip(sampled_x, sampled_y))
    random.shuffle(paired)
    sampled_x, sampled_y = zip(*paired)

    sampled_x = variable(torch.LongTensor(sampled_x).transpose(0, 1))
    sampled_y = variable(torch.FloatTensor(sampled_y).view(-1, 2))
    prevalence_var = variable(torch.FloatTensor([prevalence, 1 - prevalence]).view([1, 2]))

    return sampled_x, sampled_y, prevalence_var

def sample_data_(x_pos, x_neg, prevalence, batch_size):
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


embedding_size = 100

class_lstm_hidden_size = 128
class_lstm_layers = 1
class_lin_layers_sizes = [64,32]
dropout = 0.2

class_loss_function = torch.nn.MSELoss()


def accuracy(y_hard_true, y_soft_pred):
    pred = y_soft_pred[:, 0] > 0.5
    truth = y_hard_true[:, 0] > 0.5
    return torch.mean((pred == truth).type(torch.FloatTensor)).data[0]


class_steps = 20000
status_every = 100
test_every = 1000
save_every = 1000


def get_name(step):
    filename = 'net_' + str(step)
    return filename + '.pt'


class_net = LSTMTextClassificationNet(max_features, embedding_size, classes, class_lstm_hidden_size,
                                      class_lstm_layers, class_lin_layers_sizes, dropout)

if use_cuda:
    class_net = class_net.cuda()

print(class_net)

lr = 0.0001
weight_decay = 0.0001
prevalence = 0.5
batch_size = 1000
class_optimizer = torch.optim.Adam(class_net.parameters(), lr=lr, weight_decay=weight_decay)

x_train_pos = x_train[y_train==1]
x_train_neg = x_train[y_train!=1]

with open('class_net_hist.txt', mode='w', encoding='utf-8') as outputfile, \
        open('class_net_test.txt', mode='w', encoding='utf-8') as testoutputfile:
    class_loss_sum, quant_loss_sum, acc_sum = 0, 0, 0
    t_init = time()
    for step in range(1, class_steps + 1):

        x, y_class, y_quant = sample_data_(x_train_pos, x_train_neg, prevalence, batch_size)

        class_optimizer.zero_grad()

        class_net.train()

        y_class_pred = class_net.forward(x)
        class_loss = class_loss_function(y_class_pred, y_class)

        class_loss.backward()

        class_optimizer.step()

        class_loss_sum += class_loss.data[0]
        acc_sum += accuracy(y_class, y_class_pred)

        if step % status_every == 0:
            print('step {}\tloss {:.5f}\t acc {:.5f}\t v {:.2f} steps/s'.format(step, class_loss_sum / status_every,
                                                                                acc_sum / status_every, status_every/(time()-t_init)))
            # print(f'step {step} class_loss {class_loss_sum / status_every:.5}',
            #       f'class_acc {acc_sum / status_every:.3}')
            # print(f'step {step} class_loss {class_loss_sum / status_every:.5}',
            #       f'class_acc {acc_sum / status_every:.3}',
            #       file=outputfile)
            class_loss_sum, acc_sum = 0, 0
            t_init = time()

        if step % test_every == 0:
            class_net.eval()
            test_var_x, test_var_y, y_quant = sample_data(x_test, y_test, prevalence, batch_size)
            y_class_pred = class_net.forward(test_var_x)
            test_accuracy = accuracy(test_var_y, y_class_pred)
            print('testacc {:.5f}'.format(test_accuracy))
            # print(f'step {step}',
            #       f'test_class_acc {test_accuracy:.3}')
            # print(f'step {step}',
            #       f'test_class_acc {test_accuracy:.3}', file=testoutputfile)
        if step % save_every == 0:
            filename = get_name(step)
            print('saving to', filename)
            with open('class_' + filename, mode='bw') as modelfile:
                torch.save(class_net, modelfile)
