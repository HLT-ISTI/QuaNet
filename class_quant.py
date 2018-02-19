import random
import time
import torch
import torch.nn.functional as F
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np
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
output_size = 2

use_cuda = True

def choices(list, k):
    return np.random.permutation(list)[:k]

def variable(tensor):
    var = torch.autograd.Variable(tensor)
    return var.cuda() if use_cuda else var

def sample_data(x, y, prevalence, min_sample_length=min_sample_length, max_sample_length=max_sample_length, use_cuda=True):
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
    sampled_y = variable(torch.LongTensor(sampled_y))
    prevalence = variable(torch.FloatTensor([prevalence]))

    return sampled_x, sampled_y, prevalence


embedding_size = 200
lstm_hidden_size = 128
lstm_layers = 1
lin_layers_sizes = [128]#[128, 128]


class MyNet(torch.nn.Module):
    def __init__(self, item_count, embedding_size, lstm_hidden_size, lstm_layers, lin_layers_sizes):
        super().__init__()
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.item_emb = torch.nn.Embedding(item_count, embedding_size)
        self.item_lstm = torch.nn.LSTM(embedding_size, lstm_hidden_size, lstm_layers)
        prev_size = lstm_hidden_size
        self.item_lins = torch.nn.ModuleList()
        for lin_size in lin_layers_sizes:
            self.item_lins.append(torch.nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.class_output = torch.nn.Linear(prev_size, 1)

        self.lstm_set_hidden_size = self.lstm_hidden_size // 4
        self.classout2hidden = torch.nn.Linear(1, self.lstm_set_hidden_size)  # conditioning on class_out
        self.set_lstm = torch.nn.LSTM(self.lstm_set_hidden_size, self.lstm_set_hidden_size, lstm_layers)
        prev_size = self.lstm_set_hidden_size
        self.set_lins = torch.nn.ModuleList()
        for lin_size in lin_layers_sizes:
            self.set_lins.append(torch.nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.output = torch.nn.Linear(prev_size, 1)

    def init_class_hidden(self, batch_size):
        return (variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size)),
                variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size)))

    def init_quant_hidden(self):
        return (variable(torch.zeros(self.lstm_layers, 1, self.lstm_set_hidden_size)),
                variable(torch.zeros(self.lstm_layers, 1, self.lstm_set_hidden_size)))

    def forward(self, x):
        # document soft-classification
        embedded = self.item_emb(x)
        rnn_output, rnn_hidden = self.item_lstm(embedded, self.init_class_hidden(x.size()[1]))
        abstracted = rnn_hidden[0][-1]
        sample_size = abstracted.size()[0]
        abstracted = abstracted.view([sample_size, 1, self.lstm_hidden_size])
        for linear in self.item_lins:
            abstracted = F.relu(linear(abstracted))
        output = self.class_output(abstracted)
        class_output = F.sigmoid(output.view([sample_size, 1]))

        # quantification
        setlstm_input = self.classout2hidden(class_output)
        rnn_output, rnn_hidden = self.set_lstm(setlstm_input.view([sample_size,1,self.lstm_set_hidden_size]), self.init_quant_hidden())
        abstracted = rnn_hidden[0][-1]
        for linear in self.set_lins:
            abstracted = F.relu(linear(abstracted))
        quant_output = F.sigmoid(self.output(abstracted))

        return class_output, quant_output

net = MyNet(max_features, embedding_size, lstm_hidden_size, lstm_layers, lin_layers_sizes)
if use_cuda:
    net.cuda()

print(net)

lr = 0.0001

# optimizer = torch.optim.SGD(net.parameters(), lr=lr)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

loss_function = torch.nn.MSELoss()
class_loss_function = torch.nn.MSELoss()#torch.nn.CrossEntropyLoss()

#prevalence = random.random()
prevalence = 0.5
test_x, test_y_class, test_y_quant = sample_data(x_test, y_test, prevalence, use_cuda=use_cuda)

#prevalence = 0.5
#max_class_loss_value = 0

# classification loss must go down 75% wrt its max value
# before starting using the quantification loss
#wait_factor = 0.75

def accuracy(y_hard_true, y_soft_pred):
    return torch.sum(y_hard_true[y_soft_pred.data.view(-1) > 0.5]).float() / y_hard_true.size()[0]

steps = 1000000
print_every = 1000
start = time.time()
show_steps = 10
clip_value=None # 0.25
with open('hist.txt', mode='a', encoding='utf-8') as outfile:
    loss_ave, quantloss_ave, acc_ave  = 0, 0, 0
    for step in range(1,steps+1):
        x, y_class, y_quant = sample_data(x_train, y_train, prevalence, use_cuda=use_cuda)

        optimizer.zero_grad()

        y_class_pred, y_quant_pred = net.forward(x)
        quant_loss = loss_function(y_quant_pred, y_quant.view([1,1])) #batch=1,outs=1
        loss = quant_loss #class_loss + quant_loss

        loss.backward()
        if clip_value:
            clip = 0.25
            torch.nn.utils.clip_grad_norm(net.parameters(), clip)
            for p in net.parameters():
                p.data.add_(-lr, p.grad.data)
        else:
            optimizer.step()

        loss_ave += loss.data[0]
        quantloss_ave += quant_loss.data[0]

        acc_ave += accuracy(y_class,y_class_pred)

        if step % show_steps == 0:
            print('step=%d loss=%.5f [acc=%.2f%% quantloss=%.5f]' % (step, loss_ave/show_steps, 100*acc_ave/show_steps, quantloss_ave/show_steps))
            loss_ave, quantloss_ave, acc_ave = 0, 0, 0

        if step % print_every == 0:
            # with open('net_' + str(step) + '.pkl', mode='bw') as modelfile:
            #     torch.save(net, modelfile)

            y_class_pred, y_quant_pred = net.forward(test_x)
            print('classification', accuracy(test_y_class, y_class_pred))
            print('quantification', test_y_quant, y_quant_pred)

            start = time.time()

        # the range of possible prevalence values gets biggers as classification loss improves
#        prevalence_range = min(0.5, (max_class_loss_value - class_loss_value * wait_factor) / max_class_loss_value)
 #       prevalence = 0.5 + random.random() * prevalence_range - prevalence_range / 2
        prevalence = 0.5+(random.random()/4)
        #print(prevalence)

