import torch
import torch.nn.functional as F


class LSTMQuantificationNet(torch.nn.Module):
    def __init__(self, classes, inputsize, quant_lstm_hidden_size, quant_lstm_layers, quant_lin_layers_sizes, bidirectional=True,
                 stats_in_sequence=False, stats_in_lin_layers=False, drop_p=0.5):
        super().__init__()

        self.quant_lstm_hidden_size = quant_lstm_hidden_size
        self.quant_lstm_layers = quant_lstm_layers

        self.stats_in_sequence = stats_in_sequence
        self.stats_in_lin_layers = stats_in_lin_layers
        self.drop_p = drop_p

        # self.classout2hidden = torch.nn.Linear(classes, self.quant_lstm_hidden_size)
        # self.quant_lstm = torch.nn.LSTM(quant_lstm_hidden_size, quant_lstm_hidden_size, quant_lstm_layers)
        self.bidirectional = bidirectional
        self.quant_lstm = torch.nn.LSTM(inputsize, quant_lstm_hidden_size, quant_lstm_layers, bidirectional=bidirectional)
        prev_size = self.quant_lstm_hidden_size * (2 if bidirectional else 1)
        stats_size = ((6 * 2) if stats_in_lin_layers else 0)  # number of statistics used alongside the last dense layer
        prev_size += stats_size
        self.set_lins = torch.nn.ModuleList()
        for lin_size in quant_lin_layers_sizes:
            self.set_lins.append(torch.nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.quant_output = torch.nn.Linear(prev_size, classes)

    def init_quant_hidden(self, batch_size):
        directions = 2 if self.bidirectional else 1
        var_hidden = torch.autograd.Variable(
            torch.zeros(self.quant_lstm_layers * directions, batch_size, self.quant_lstm_hidden_size))
        var_cell = torch.autograd.Variable(
            torch.zeros(self.quant_lstm_layers * directions, batch_size, self.quant_lstm_hidden_size))
        if next(self.quant_lstm.parameters()).is_cuda:
            return (var_hidden.cuda(), var_cell.cuda())
        else:
            return (var_hidden, var_cell)

    def forward(self, x, stats=None):
        # lstm_input = self.classout2hidden(x)
        if not stats is None and self.stats_in_sequence:
            lstm_input = torch.cat([stats, x])
        else:
            lstm_input = x
        lstm_input = lstm_input.transpose(0, 1)
        rnn_output, rnn_hidden = self.quant_lstm(lstm_input, self.init_quant_hidden(x.size()[0]))
        abstracted = rnn_output[-1]  # rnn_hidden[0][-1]

        if not stats is None and self.stats_in_lin_layers:
            abstracted = torch.cat([stats.view(abstracted.shape[0],-1), abstracted], dim=1)

        for linear in self.set_lins:
            abstracted = F.dropout(F.relu(linear(abstracted)), self.drop_p, self.training)


        quant_output = F.softmax(self.quant_output(abstracted), dim=1)
        # quant_output = self.quant_output(abstracted)
        # if not self.training:
        #     quant_output = torch.clamp(quant_output, 0, 1)

        return quant_output
