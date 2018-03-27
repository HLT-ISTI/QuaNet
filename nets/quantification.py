import torch
import torch.nn.functional as F


class LSTMQuantificationNet(torch.nn.Module):
    def __init__(self, classes, quant_lstm_hidden_size, quant_lstm_layers, quant_lin_layers_sizes):
        super().__init__()

        self.quant_lstm_hidden_size = quant_lstm_hidden_size
        self.quant_lstm_layers = quant_lstm_layers

        #self.classout2hidden = torch.nn.Linear(classes, self.quant_lstm_hidden_size)
        #self.quant_lstm = torch.nn.LSTM(quant_lstm_hidden_size, quant_lstm_hidden_size, quant_lstm_layers)
        self.quant_lstm = torch.nn.LSTM(classes, quant_lstm_hidden_size, quant_lstm_layers)
        prev_size = self.quant_lstm_hidden_size
        self.set_lins = torch.nn.ModuleList()
        for lin_size in quant_lin_layers_sizes:
            self.set_lins.append(torch.nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.quant_output = torch.nn.Linear(prev_size, classes)

    def init_quant_hidden(self, batch_size):
        var_hidden = torch.autograd.Variable(
            torch.zeros(self.quant_lstm_layers, batch_size, self.quant_lstm_hidden_size))
        var_cell = torch.autograd.Variable(torch.zeros(self.quant_lstm_layers, batch_size, self.quant_lstm_hidden_size))
        if next(self.quant_lstm.parameters()).is_cuda:
            return (var_hidden.cuda(), var_cell.cuda())
        else:
            return (var_hidden, var_cell)

    def forward(self, x):
        #lstm_input = self.classout2hidden(x)
        lstm_input = x
        lstm_input = lstm_input.transpose(0, 1)
        rnn_output, rnn_hidden = self.quant_lstm(lstm_input, self.init_quant_hidden(x.size()[0]))
        abstracted = rnn_output[-1]# rnn_hidden[0][-1]
        for linear in self.set_lins:
            abstracted = F.relu(linear(abstracted))
        quant_output = F.softmax(self.quant_output(abstracted), dim=1)

        return quant_output
