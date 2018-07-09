import torch
import torch.nn.functional as F
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 1000
maxlen = 80
batch_size = 500
embedding_dims = 25
hidden_dims = 25
epochs = 50
dropout = 0.2

loss_function = torch.nn.BCELoss()

### IMDB is a too simple dataset, using it just to see it this idea works

print('Loading data...')
# start_char is set to 2 because 0 is for padding and 1 is for sequence restart signal
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, start_char=2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Repeated sequence data

import numpy as np

# using the reserved id 1 as the signal of sequence restart
d_x_train = np.hstack([x_train, np.expand_dims(np.asarray([1] * x_train.shape[0]), 1), x_train])
d_x_test = np.hstack([x_test, np.expand_dims(np.asarray([1] * x_test.shape[0]), 1), x_test])


# support functions

def sample_batch(x, y, batch, batch_size):
    batch_start = batch * batch_size
    batch_end = batch_start + batch_size
    batch_x = x[batch_start:batch_end]
    batch_y = y[batch_start:batch_end]

    return torch.LongTensor(batch_x).transpose(0, 1).cuda(), torch.FloatTensor(batch_y).cuda()


def accuracy(y_hard_true, y_soft_pred):
    pred = y_soft_pred[:, 0] > 0.5
    truth = y_hard_true > 0.5
    return sum(pred.tolist() == truth) / len(truth)


def fit(x_train, y_train, x_test, y_test, net, optimizer, loss_function, batch_size, epochs):
    batches = len(x_train) // batch_size
    results = list()
    for epoch in range(epochs):
        for batch in range(batches):
            x, y = sample_batch(x_train, y_train, batch, batch_size)

            optimizer.zero_grad()

            net.train()

            y_hat = net(x)

            loss = loss_function(y_hat, y)

            loss.backward()

            optimizer.step()

        net.eval()

        y_test_hat = net(torch.LongTensor(x_test).transpose(0, 1).cuda())
        results.append((epoch, loss.item(), accuracy(y_test, y_test_hat)))
    return results


# LSTM

class LSTMNet(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_size, classes, lstm_hidden_size,
                 dropout, bidirectional=False):
        super().__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = torch.nn.LSTM(embedding_size, lstm_hidden_size, 1, bidirectional=bidirectional)
        if self.bidirectional:
            self.linear = torch.nn.Linear(lstm_hidden_size * 2, classes)
        else:
            self.linear = torch.nn.Linear(lstm_hidden_size, classes)

    def init_lstm_hidden(self, set_size):
        first_dimension = 1
        if self.bidirectional:
            first_dimension = 2
        hidden = torch.zeros(first_dimension, set_size, self.lstm_hidden_size)
        cell = torch.zeros(first_dimension, set_size, self.lstm_hidden_size)
        if next(self.lstm.parameters()).is_cuda:
            return (hidden.cuda(), cell.cuda())
        else:
            return (hidden, cell)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, lstm_hidden = self.lstm(embedded, self.init_lstm_hidden(x.size()[1]))
        if self.bidirectional:
            output = F.dropout(
                torch.cat((lstm_output[-1, :, :self.lstm_hidden_size], lstm_output[0, :, self.lstm_hidden_size:]), 1),
                self.dropout, self.training)
        else:
            output = F.dropout(lstm_output[-1, :, :], self.dropout, self.training)

        return F.sigmoid(self.linear(output))


# Repeat LSTM

class RepeatLSTMNet(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_size, classes, lstm_hidden_size,
                 dropout, bidirectional=False):
        super().__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = torch.nn.LSTM(embedding_size, lstm_hidden_size, 1, bidirectional=bidirectional)
        self.embedding_repeat = torch.nn.Embedding(vocabulary_size, embedding_size)
        self.lstm_repeat = torch.nn.LSTM(embedding_size, lstm_hidden_size, 1, bidirectional=bidirectional)
        if self.bidirectional:
            self.linear = torch.nn.Linear(lstm_hidden_size * 2, classes)
        else:
            self.linear = torch.nn.Linear(lstm_hidden_size, classes)

    def init_lstm_hidden(self, set_size):
        first_dimension = 1
        if self.bidirectional:
            first_dimension = 2
        hidden = torch.zeros(first_dimension, set_size, self.lstm_hidden_size)
        cell = torch.zeros(first_dimension, set_size, self.lstm_hidden_size)
        if next(self.lstm.parameters()).is_cuda:
            return (hidden.cuda(), cell.cuda())
        else:
            return (hidden, cell)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, lstm_hidden = self.lstm(embedded, self.init_lstm_hidden(x.size()[1]))
        embedded_repeat = self.embedding_repeat(x)
        lstm_output, lstm_hidden = self.lstm_repeat(embedded_repeat, lstm_hidden)
        if self.bidirectional:
            output = F.dropout(
                torch.cat((lstm_output[-1, :, :self.lstm_hidden_size], lstm_output[0, :, self.lstm_hidden_size:]), 1),
                self.dropout)
        else:
            output = F.dropout(lstm_output[-1, :, :], self.dropout)

        return F.sigmoid(self.linear(output))


results = list()

runs = 10

for run in range(runs):

    # Simple LSTM

    net = LSTMNet(max_features, embedding_dims, 1, hidden_dims, dropout)
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters())

    results.append(('LSTM', fit(x_train, y_train, x_test, y_test, net, optimizer, loss_function, batch_size, epochs)))

    # Bidirectional LSTM

    net = LSTMNet(max_features, embedding_dims, 1, hidden_dims, dropout, bidirectional=True)
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters())

    results.append(('BiLSTM', fit(x_train, y_train, x_test, y_test, net, optimizer, loss_function, batch_size, epochs)))

    # Repeated sequence + Simple LSTM

    net = LSTMNet(max_features, embedding_dims, 1, hidden_dims, dropout)
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters())

    results.append(
        ('2S-LSTM', fit(d_x_train, y_train, d_x_test, y_test, net, optimizer, loss_function, batch_size, epochs)))

    # Repeated sequence + Bidirectional LSTM

    net = LSTMNet(max_features, embedding_dims, 1, hidden_dims, dropout, bidirectional=True)
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters())

    results.append(
        ('2S-BiLSTM', fit(d_x_train, y_train, d_x_test, y_test, net, optimizer, loss_function, batch_size, epochs)))

    # Repeat LSTM

    net = RepeatLSTMNet(max_features, embedding_dims, 1, hidden_dims, dropout)
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters())

    results.append(('ReLSTM', fit(x_train, y_train, x_test, y_test, net, optimizer, loss_function, batch_size, epochs)))

    # Repeat Bidirectional LSTM

    net = RepeatLSTMNet(max_features, embedding_dims, 1, hidden_dims, dropout, bidirectional=True)
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters())

    results.append(
        ('ReBiLSTM', fit(x_train, y_train, x_test, y_test, net, optimizer, loss_function, batch_size, epochs)))

print('last epoch')
for method, epoch_results in results:
    print(method, epoch_results[-1])

print('best epoch')
for method, epoch_results in results:
    max_result = list(sorted(epoch_results, key=lambda x: x[2]))[-1]
    print(method, max_result)
