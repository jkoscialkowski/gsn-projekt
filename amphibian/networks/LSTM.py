# TODO: test if works

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, batch_size, seq_len, input_size, hidden_size, n_outputs,
                 num_layers=1, dropout=0.1): # remember about cuda_device
        """
        Class LSTMModel - implementation of simple LSTM architecture

        :param batch_size: size of the batch
        :param seq_len: number of days
        :param input_size: number of inputs in the specific day
        :param hidden_size: number of neurons in the
        :param n_outputs: number of output values from the fully connected layer
        :param num_layers: number of layers of the RNN
        :param dropout: dropout probability in the LSTM layer
        """
        super().__init__()
        # Setting parameters
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.n_outputs)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, X):
        # remember about transforming data in the right dimension - (seq_len X batch_size X input_size)
        hidden = self.init_hidden()
        out, _ = self.lstm(X, hidden)
        out = self.fc(out[-1, :, :].squeeze())
        return out


