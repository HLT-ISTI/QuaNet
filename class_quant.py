import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 5000
max_len = 120

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
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
    sampled_y.extend([1] * sample_pos_count)
    sampled_y.extend([0] * sample_neg_count)

    paired = list(zip(sampled_x, sampled_y))
    random.shuffle(paired)
    sampled_x, sampled_y = zip(*paired)

    sampled_x = variable(torch.LongTensor(sampled_x).transpose(0, 1))
    sampled_y = variable(torch.FloatTensor(sampled_y).view(-1, 1))
    prevalence = variable(torch.FloatTensor([prevalence]).view([1, 1]))

    return sampled_x, sampled_y, prevalence


class MyNet(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_size, class_lstm_hidden_size, class_lstm_layers,
                 class_lin_layers_sizes,
                 quant_lstm_hidden_size, quant_lstm_layers, quant_lin_layers_sizes):
        super().__init__()

        # classification part
        self.class_lstm_layers = class_lstm_layers
        self.class_lstm_hidden_size = class_lstm_hidden_size

        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_size)
        self.class_lstm = torch.nn.LSTM(embedding_size, class_lstm_hidden_size, class_lstm_layers)
        prev_size = class_lstm_hidden_size
        self.class_lins = torch.nn.ModuleList()
        for lin_size in class_lin_layers_sizes:
            self.class_lins.append(torch.nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.class_output = torch.nn.Linear(prev_size, 1)

        # quantification part
        self.quant_lstm_hidden_size = quant_lstm_hidden_size
        self.quant_lstm_layers = quant_lstm_layers

        self.classout2hidden = torch.nn.Linear(1, self.quant_lstm_hidden_size)  # conditioning on class_out
        self.quant_lstm = torch.nn.LSTM(quant_lstm_hidden_size, quant_lstm_hidden_size, quant_lstm_layers)
        prev_size = self.quant_lstm_hidden_size
        self.set_lins = torch.nn.ModuleList()
        for lin_size in quant_lin_layers_sizes:
            self.set_lins.append(torch.nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.quant_output = torch.nn.Linear(prev_size, 1)

    def init_class_hidden(self, set_size):
        return (variable(torch.zeros(self.class_lstm_layers, set_size, self.class_lstm_hidden_size)),
                variable(torch.zeros(self.class_lstm_layers, set_size, self.class_lstm_hidden_size)))

    def init_quant_hidden(self, batch_size):
        return (variable(torch.zeros(self.quant_lstm_layers, batch_size, self.quant_lstm_hidden_size)),
                variable(torch.zeros(self.quant_lstm_layers, batch_size, self.quant_lstm_hidden_size)))

    def forward_class(self, x):
        # classification
        embedded = self.embedding(x)
        rnn_output, rnn_hidden = self.class_lstm(embedded, self.init_class_hidden(x.size()[1]))
        abstracted = rnn_hidden[0][-1]
        abstracted = abstracted.unsqueeze(-2)
        for linear in self.class_lins:
            abstracted = F.relu(linear(abstracted))
        output = self.class_output(abstracted)
        class_output = F.sigmoid(output)
        return class_output

    def forward_quant(self, x):
        # quantification
        lstm_input = self.classout2hidden(x)
        lstm_input = lstm_input.transpose(0,1)
        rnn_output, rnn_hidden = self.quant_lstm(lstm_input, self.init_quant_hidden(x.size()[0]))
        abstracted = rnn_hidden[0][-1]
        for linear in self.set_lins:
            abstracted = F.relu(linear(abstracted))
        quant_output = F.sigmoid(self.quant_output(abstracted))

        return quant_output


embedding_size = 200

class_lstm_hidden_size = 128
class_lstm_layers = 1
class_lin_layers_sizes = [128, 64]

quant_lstm_hidden_size = 32
quant_lstm_layers = 1
quant_lin_layers_sizes = [32, 16]

class_loss_function = torch.nn.MSELoss()  # torch.nn.CrossEntropyLoss()
quant_loss_function = torch.nn.MSELoss()


def classify_and_count(y_soft_pred):
    pred = y_soft_pred > 0.5
    acc = torch.mean(pred.type(torch.FloatTensor)).data[0]
    return acc


def accuracy(y_hard_true, y_soft_pred):
    pred = y_soft_pred > 0.5
    truth = y_hard_true > 0.5
    acc = torch.mean((pred == truth).type(torch.FloatTensor)).data[0]
    return acc


steps = 20000
status_every = 20
test_every = 100
save_every = 1000

starting_step = 0
stop_teacher_step = 10000
end_to_end_step = steps + 1

use_teacher = starting_step <= stop_teacher_step
split_learning = starting_step <= end_to_end_step


def get_name(step, split_learning, use_teacher, stop_teacher_step, end_to_end_step):
    filename = 'net_' + str(step)
    if split_learning:
        filename += '_split'
        if use_teacher:
            filename += '_teacher'
        else:
            filename += '_noteacher-' + str(stop_teacher_step)
    else:
        filename += '_endtoend-' + str(end_to_end_step)
    return filename + '.pt'


if starting_step > 0:
    filename = get_name(starting_step, split_learning, use_teacher, stop_teacher_step, end_to_end_step)
    with open(filename, mode='br') as modelfile:
        net = torch.load(modelfile)
else:
    net = MyNet(max_features, embedding_size, class_lstm_hidden_size, class_lstm_layers, class_lin_layers_sizes,
                quant_lstm_hidden_size, quant_lstm_layers, quant_lin_layers_sizes)
if use_cuda:
    net.cuda()

print(net)

lr = 0.0001

optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

quant_shuffles = 100

with open('hist_' + str(time.time()) + '.txt', mode='w', encoding='utf-8') as outputfile, \
        open('test' + str(time.time()) + '.txt', mode='w', encoding='utf-8') as testoutputfile:
    class_loss_sum, quant_loss_sum, acc_sum = 0, 0, 0
    for step in range(starting_step + 1, steps + 1):
        if step > stop_teacher_step:
            use_teacher = False

        prevalence = random.random()
        x, y_class, y_quant = sample_data(x_train, y_train, prevalence)

        optimizer.zero_grad()

        y_class_pred = net.forward_class(x)
        class_loss = class_loss_function(y_class_pred, y_class)
        if split_learning:
            y_class_perms = list()
            y_quant_repeat = list()
            if use_teacher:
                y_class_to_repeat = y_class
            else:
                y_class_to_repeat = y_class_pred.squeeze(-1)
            for _ in range(quant_shuffles):
                y_class_perms.append(y_class_to_repeat.data[torch.cuda.LongTensor(np.random.permutation(len(y_class)).tolist())].unsqueeze(0))
                y_quant_repeat.append((y_quant.data))
            y_class_perms = variable(torch.cat(y_class_perms))
            y_quant = variable(torch.cat(y_quant_repeat))

            y_quant_pred = net.forward_quant(y_class_perms)
            quant_loss = quant_loss_function(y_quant_pred, y_quant)

            class_loss.backward()
            quant_loss.backward()
        else:
            y_quant_pred = net.forward_quant(y_class_pred)
            quant_loss = quant_loss_function(y_quant_pred.unsqueeze(0), y_quant)

            quant_loss.backward()

        optimizer.step()

        if split_learning:
            class_loss_sum += class_loss.data[0]
        quant_loss_sum += quant_loss.data[0]
        acc_sum += accuracy(y_class, y_class_pred)

        if step % status_every == 0:
            print('step=%d class_loss=%.5f quant_loss=%.5f class_acc=%.2f' % (
                step, class_loss_sum / status_every, quant_loss_sum / status_every, 100 * acc_sum / status_every))
            print('step=%d class_loss=%.5f quant_loss=%.5f class_acc=%.2f' % (
                step, class_loss_sum / status_every, quant_loss_sum / status_every, 100 * acc_sum / status_every),
                  file=outputfile)
            class_loss_sum, quant_loss_sum, acc_sum = 0, 0, 0

        if step % test_every == 0:
            num_test = 5
            for _ in range(num_test):
                prevalence = random.random()
                test_x, test_y_class, test_y_quant = sample_data(x_test, y_test, prevalence)
                y_class_pred = net.forward_class(test_x)
                y_quant_pred = net.forward_quant(test_y_class.unsqueeze(0))
                real_y_quant_pred = net.forward_quant(y_class_pred)
                print('step', step, 'split', split_learning, 'teacher', use_teacher, 'class_acc',
                      accuracy(test_y_class, y_class_pred),
                      'true_prev', test_y_quant.data[0][0], 'class_prev', classify_and_count(y_class_pred),
                      'pred_teacher_prev', y_quant_pred.data[0][0], 'pred_end_to_end_prev',
                      real_y_quant_pred.data[0][0])
                print('step', step, 'split', split_learning, 'teacher', use_teacher, 'class_acc',
                      accuracy(test_y_class, y_class_pred),
                      'true_prev', test_y_quant.data[0][0], 'class_prev', classify_and_count(y_class_pred),
                      'pred_teacher_prev', y_quant_pred.data[0][0], 'pred_end_to_end_prev',
                      real_y_quant_pred.data[0][0], file=testoutputfile)

        if step % save_every == 0:
            filename = get_name(step, split_learning, use_teacher, stop_teacher_step, end_to_end_step)
            print('saving to', filename)
            with open(filename, mode='bw') as modelfile:
                torch.save(net, modelfile)
