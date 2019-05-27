# TODO: test if works

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModel(nn.Module):
    def __init__(self, batch_size, seq_len, input_size, hidden_size, n_outputs,
                 num_layers=1, dropout=0.1, recurrent_type='rnn',
                 dotprod_ffnn=False):
        """
        Class AttentionModel - implementation of simple Attention architecture

        :param batch_size: size of the batch
        :param seq_len: number of days
        :param input_size: number of inputs in the specific day
        :param hidden_size: number of neurons in the
        :param n_outputs: number of output values from the fully connected layer
        :param num_layers: number of layers of the RNN
        :param dropout: dropout probability in the LSTM layer
        :param recurrent_type: whether to use RNN or LSTM as recurrent layers
        :param dotprod_ffnn: whether to use dot product or feedforward NN
        """
        super().__init__()
        assert recurrent_type in ['rnn', 'lstm']
        # Setting parameters
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.w_a = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.u_a = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.v_a = nn.Parameter(torch.randn(hidden_size, 1))

        if recurrent_type == 'rnn':
            self.recurrent_pre = nn.RNN(input_size=self.input_size,
                                        hidden_size=self.hidden_size,
                                        num_layers=self.num_layers,
                                        dropout=self.dropout)
            self.recurrent_cell_post = nn.RNNCell(input_size=self.hidden_size,
                                                  hidden_size=self.hidden_size)
        elif recurrent_type == 'lstm':
            self.recurrent_pre = nn.LSTM(input_size=self.input_size,
                                         hidden_size=self.hidden_size,
                                         num_layers=self.num_layers,
                                         dropout=self.dropout)
            self.recurrent_cell_post = nn.LSTMCell(input_size=self.input_size,
                                                   hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.n_outputs)

    def init_hidden(self):
        if self.recurrent_type == 'rnn':
            return torch.zeros(self.num_layers,
                               self.batch_size,
                               self.hidden_size)
        elif self.recurrent_type == 'lstm':
            return (torch.zeros(self.num_layers,
                                self.batch_size,
                                self.hidden_size),
                    torch.zeros(self.num_layers,
                                self.batch_size,
                                self.hidden_size))

    def forward(self, X):
        # remember about transforming data in the right dimension - (seq_len X batch_size X input_size)
        # Initialize first hidden state for the pre-RNN with zeros
        hidden = self.init_hidden()
        out_pre, _ = self.recurrent_pre(X, hidden)

        # Initialize input to post-RNN with zeros
        hidden_post = torch.zeros(self.batch_size, self.hidden_size)

        for el in range(self.hidden_size):
            if self.dotprod_ffnn:
                # Mix last hidden state of post-RNN and all hidden states of
                # pre-RNN using a feedforward NN
                pre_soft = out_pre.matmul(self.w_a) \
                           + hidden_post.matmul(self.u_a)
                pre_soft = torch.tanh(pre_soft).matmul(v_a).squeeze()
            else:
                # Dot products of last hidden state of post-RNN and
                # all hidden states of pre-RNN
                pre_soft = torch.sum(out_pre * hidden_post, 2)
            # Softmaxing
            post_soft = F.softmax(pre_soft, dim=0)
            # Applying softmax probabilities to corresponding hidden pre-RNN states
            attention = out_pre.permute(2, 0, 1) * post_soft
            # Summing softmax-scaled pre-RNN hidden states and transposing
            attention = torch.sum(attention, 1).t()
            hidden_post = self.rnn_cell_post(hidden_post, attention)
        out = self.fc(hidden_post)
        return out

