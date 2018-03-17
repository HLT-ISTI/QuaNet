import torch
import torch.nn.functional as F


class LSTMTextClassificationNet(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_size, classes, class_lstm_hidden_size, class_lstm_layers,
                 class_lin_layers_sizes, dropout):
        super().__init__()

        self.class_lstm_layers = class_lstm_layers
        self.class_lstm_hidden_size = class_lstm_hidden_size

        self.dropout = dropout

        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_size)
        self.class_lstm = torch.nn.LSTM(embedding_size, class_lstm_hidden_size, class_lstm_layers)
        prev_size = class_lstm_hidden_size
        self.class_lins = torch.nn.ModuleList()
        for lin_size in class_lin_layers_sizes:
            self.class_lins.append(torch.nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.class_output = torch.nn.Linear(prev_size, classes)

    def init_class_hidden(self, set_size):
        var_hidden = torch.autograd.Variable(torch.zeros(self.class_lstm_layers, set_size, self.class_lstm_hidden_size))
        var_cell = torch.autograd.Variable(torch.zeros(self.class_lstm_layers, set_size, self.class_lstm_hidden_size))
        if next(self.class_lstm.parameters()).is_cuda:
            return (var_hidden.cuda(), var_cell.cuda())
        else:
            return (var_hidden, var_cell)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_output, rnn_hidden = self.class_lstm(embedded, self.init_class_hidden(x.size()[1]))
        abstracted = F.dropout(rnn_hidden[0][-1], self.dropout)
        for linear in self.class_lins:
            abstracted = F.dropout(F.relu(linear(abstracted)))
        output = self.class_output(abstracted)
        class_output = F.softmax(output)
        return class_output


class CNNTextClassificationNet(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_size, classes, class_cnn_filter_sizes, class_cnn_filter_counts,
                 class_lin_layers_sizes):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_size)

        self.cnns = torch.nn.ModuleList()

        for filter_size, filter_count in zip(class_cnn_filter_sizes, class_cnn_filter_counts):
            self.cnns.append(torch.nn.Conv2d(1, filter_count, (filter_size, embedding_size)))

        self.class_lins = torch.nn.ModuleList()
        prev_size = class_lin_layers_sizes[0]
        for lin_size in class_lin_layers_sizes[1:]:
            self.class_lins.append(torch.nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.class_output = torch.nn.Linear(prev_size, classes)

    def init_class_hidden(self, set_size):
        var_hidden = torch.zeros(self.class_lstm_layers, set_size, self.class_lstm_hidden_size)
        var_cell = torch.zeros(self.class_lstm_layers, set_size, self.class_lstm_hidden_size)
        if self.is_cuda:
            return (var_hidden.cuda(), var_cell.cuda())
        else:
            return (var_hidden, var_cell)

    def forward(self, x):
        embedded = self.embedding(x)

        convoluted = embedded.unsqueeze(1)
        for cnn in self.cnns:
            convoluted = F.relu(cnn(convoluted)).squeeze(3)

        abstracted = F.adaptive_avg_pool1d(convoluted, self.class_lins[0].size()[0])
        for linear in self.class_lins:
            abstracted = F.relu(linear(abstracted))
        output = self.class_output(abstracted)
        class_output = F.softmax(output)
        return class_output
