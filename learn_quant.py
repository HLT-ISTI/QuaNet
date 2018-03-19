import random
import time

import numpy as np
import torch
from keras.datasets import imdb
from keras.preprocessing import sequence

from nets.classification import LSTMTextClassificationNet
from nets.quantification import LSTMQuantificationNet

max_features = 5000
max_len = 120
classes = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)


def split_train_validation(x, y, val_portion):
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

min_sample_length = 1000
max_sample_length = 1000

use_cuda = True


def choices(list, k):
    return np.random.permutation(list)[:k]


def variable(tensor):
    var = torch.autograd.Variable(tensor)
    return var.cuda() if use_cuda else var


def sample_data(x, y, prevalence, min_sample_length=min_sample_length, max_sample_length=max_sample_length):
    x_pos = x[y == 1]
    x_neg = x[y != 1]
    sample_length = random.randint(min_sample_length, max_sample_length)
    sample_pos_count = int(sample_length * prevalence)
    sample_neg_count = sample_length - sample_pos_count
    prevalence = sample_pos_count / sample_length

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


embedding_size = 200


quant_lstm_hidden_size = 32
quant_lstm_layers = 1
quant_lin_layers_sizes = [32, 16, classes]

quant_loss_function = torch.nn.MSELoss()


def classify_and_count(y_soft_pred):
    pred = y_soft_pred[:,0] > 0.5
    return torch.mean(pred.type(torch.FloatTensor)).data[0]


def accuracy(y_hard_true, y_soft_pred):
    pred = y_soft_pred[:,0] > 0.5
    truth = y_hard_true[:,0] > 0.5
    return torch.mean((pred == truth).type(torch.FloatTensor)).data[0]


def tpr(y_hard_true, y_soft_pred):
    pred = (y_soft_pred[:,0] > 0.5).type(torch.FloatTensor)
    truth = (y_hard_true[:,0] > 0.5).type(torch.FloatTensor)
    true_pos = ((pred + truth*2)==3).type(torch.FloatTensor)
    return torch.sum(true_pos).data[0] / torch.sum(truth).data[0]


def fpr(y_hard_true, y_soft_pred):
    pred = (y_soft_pred[:,0] > 0.5).type(torch.FloatTensor)
    truth = (y_hard_true[:,0] > 0.5).type(torch.FloatTensor)
    false_pos = ((pred + truth*2)==1).type(torch.FloatTensor)
    return torch.sum(false_pos).data[0] / torch.sum((truth==0).type(torch.FloatTensor)).data[0]


class_steps = 20000
quant_steps = 100000
status_every = 100
test_every = 1000
save_every = 10000


def get_name(step):
    filename = 'quant_net_' + str(step)
    return filename + '.pt'


with open('class_net_'+str(class_steps)+'.pt', mode='br') as modelfile:
    class_net = torch.load(modelfile)

quant_net = LSTMQuantificationNet(classes, quant_lstm_hidden_size, quant_lstm_layers, quant_lin_layers_sizes)

if use_cuda:
    class_net.cuda()
    quant_net.cuda()

print(class_net)
print(quant_net)

lr = 0.0001
weight_decay = 0.0001


quant_optimizer = torch.optim.Adam(quant_net.parameters(), lr=lr, weight_decay=weight_decay)
quant_shuffles = 100
test_samples = 100

class_net.eval()

with open('quant_net_hist.txt', mode='w', encoding='utf-8') as outputfile, \
        open('quant_net_test.txt', mode='w', encoding='utf-8') as testoutputfile:
    class_loss_sum, quant_loss_sum, acc_sum = 0, 0, 0
    for step in range(1, quant_steps + 1):

        prevalence = random.random()
        x, y_class, y_quant = sample_data(x_val, y_val, prevalence)

        quant_optimizer.zero_grad()

        quant_net.train()

        y_class_pred = class_net.forward(x)

        y_class_perms = list()
        y_quant_repeat = list()
        for _ in range(quant_shuffles):
            y_class_perms.append(y_class_pred.data[
                torch.cuda.LongTensor(np.random.permutation(len(y_class)).tolist())].unsqueeze(
                0))
            y_quant_repeat.append((y_quant.data))
        y_class_perms = variable(torch.cat(y_class_perms))
        y_quant = variable(torch.cat(y_quant_repeat))

        y_quant_pred = quant_net.forward(y_class_perms)
        quant_loss = quant_loss_function(y_quant_pred, y_quant)

        quant_loss.backward()

        quant_optimizer.step()

        quant_loss_sum += quant_loss.data[0]
        acc_sum += accuracy(y_class, y_class_pred)

        if step % status_every == 0:
            print(f'step {step}',
                  f'quant_loss {quant_loss_sum / status_every:.5} class_acc {acc_sum / status_every:.3}')
            print(f'step {step}',
                  f'quant_loss {quant_loss_sum / status_every:.5} class_acc {acc_sum / status_every:.3}',
                  file=outputfile)
            quant_loss_sum, acc_sum = 0, 0

        if step % test_every == 0:
            quant_net.eval()

            x_val_sample, y_val_sample, y_val_quant = sample_data(x_val, y_val, prevalence)
            y_val_pred = class_net.forward(x_val_sample)
            fpr_val = fpr(y_val_sample, y_val_pred)
            tpr_val = tpr(y_val_sample, y_val_pred)

            for prevalence in np.linspace(0, 1, 11):
                for _ in range(test_samples):
                    test_x, test_y_class, test_y_quant = sample_data(x_test, y_test, prevalence)
                    y_class_pred = class_net.forward(test_x)
                    y_quant_pred = quant_net.forward(test_y_class.unsqueeze(0))
                    real_y_quant_pred = quant_net.forward(y_class_pred.unsqueeze(0))
                    test_accuracy = accuracy(test_y_class, y_class_pred)
                    cc_prev = classify_and_count(y_class_pred)
                    cheat_prev = y_quant_pred.data[0][0]
                    net_prev = real_y_quant_pred.data[0][0]
                    if tpr_val - fpr_val !=0:
                        acc_prev = (cc_prev - fpr_val) / (tpr_val - fpr_val)
                        anet_prev = (net_prev - fpr_val) / (tpr_val - fpr_val)
                    else:
                        acc_prev = -1.
                        anet_prev = -1.
                    print(f'step {step}',
                          f'class_acc {test_accuracy:.3} true_prev {test_y_quant.data[0][0]:.3}',
                          f'cc_prev {cc_prev:.3} cheat_prev {cheat_prev:.3}',
                          f'net_prev {net_prev:.3}', f'acc_prev {acc_prev:.3}',
                          f'anet_prev {anet_prev:.3}')
                    print(f'step {step}',
                          f'class_acc {test_accuracy:.3} true_prev {test_y_quant.data[0][0]:.3}',
                          f'cc_prev {cc_prev:.3} cheat_prev {cheat_prev:.3}',
                          f'net_prev {net_prev:.3}', f'acc_prev {acc_prev:.3}',
                          f'anet_prev {anet_prev:.3}', file=testoutputfile)

        if step % save_every == 0:
            filename = get_name(step)
            print('saving to', filename)
            with open(filename, mode='bw') as modelfile:
                torch.save(class_net, modelfile)
                torch.save(quant_net, modelfile)
