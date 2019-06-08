import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Transform data in the right dimension, (seq_len X batch_size X input_size)!


class SoftmaxRegressionModel(nn.Module):
    def __init__(self, batch_size, input_size, n_outputs):
        """
        Class SoftmaxRegressionModel - implementation of Softmax Regression Model

        :param batch_size: size of the batch
        :param input_size: size of input
        :param n_outputs: size of output
        """
        super().__init__()
        # Setting parameters
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.fc = nn.Linear(in_features=self.input_size,
                            out_features=self.n_outputs)

    def forward(self, X):
        X = X.view(self.batch_size, self.input_size)
        out = self.fc(X)
        return out


class RNNModel(nn.Module):
    def __init__(self, batch_size, seq_len, input_size, hidden_size, n_outputs,
                 num_layers=1, dropout=0.1):
        """
        Class RNNModel - implementation of simple RNN model

        :param batch_size: size of the batch
        :param seq_len: number of days
        :param input_size: number of inputs in the specific day
        :param hidden_size: number of features in the hidden state
        :param n_outputs: number of output values from the fully connected layer
        :param num_layers: number of layers of the RNN
        :param dropout: dropout
        """
        super().__init__()
        # Setting parameters
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.rnn = nn.RNN(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.n_outputs)

    def init_hidden(self, ):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, X):
        hidden = self.init_hidden()
        out, _ = self.rnn(X, hidden)
        out = self.fc(out[-1, :, :].squeeze())
        return out


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
        hidden = self.init_hidden()
        out, _ = self.lstm(X, hidden)
        out = self.fc(out[-1, :, :].squeeze())
        return out


class AttentionModel(nn.Module):
    def __init__(self, batch_size, seq_len, input_size, hidden_size, n_outputs,
                 num_layers=1, dropout=0.1, recurrent_type='rnn',
                 alignment='dotprod'):
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
        :param alignment: whether to use dot product or feedforward NN
        """
        super().__init__()
        assert recurrent_type in ['rnn', 'lstm']
        assert alignment in ['dotprod', 'ffnn']
        # Setting parameters
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        # Defining alignment behaviour
        if self.alignment == 'ffnn':
            # Xavier initialisation
            self.w_a = nn.Parameter(
                torch.randn(hidden_size, hidden_size, requires_grad=True) / np.sqrt(hidden_size),
                requires_grad=True
            )
            self.u_a = nn.Parameter(
                torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
            )
            self.v_a = nn.Parameter(
                torch.randn(hidden_size, 1) / np.sqrt(hidden_size)
            )

        # Defining recurrent layers
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
            self.recurrent_cell_post = nn.LSTMCell(input_size=self.hidden_size,
                                                   hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.n_outputs)

    def init_hidden(self, which):
        assert which in ['pre', 'post']
        if which == 'pre':
            dims = self.num_layers, self.batch_size, self.hidden_size
        elif which == 'post':
            dims = self.batch_size, self.hidden_size
        return torch.zeros(*dims)

    def forward(self, X):
        # Initialize first hidden state for the pre-RNN with zeros
        if self.recurrent_type == 'rnn':
            hidden_pre = self.init_hidden('pre')
            out_pre, _ = self.recurrent_pre(X, hidden_pre)
        elif self.recurrent_type == 'lstm':
            hidden_pre, state_pre = (self.init_hidden('pre'),
                                     self.init_hidden('pre'))
            out_pre, _ = self.recurrent_pre(X, (hidden_pre, state_pre))

        # Initialize input to post-RNN with zeros
        hidden_post = self.init_hidden('post')
        if self.recurrent_type == 'lstm':
            state_post = self.init_hidden('post')
        for el in range(self.seq_len):
            if self.alignment == 'ffnn':
                # Mix last hidden state of post-RNN and all hidden states of
                # pre-RNN using a feedforward NN
                pre_soft = out_pre.matmul(self.w_a) \
                           + hidden_post.matmul(self.u_a)
                pre_soft = torch.tanh(pre_soft).matmul(self.v_a).squeeze()
            else:
                # Dot products of last hidden state of post-RNN and
                # all hidden states of pre-RNN
                pre_soft = torch.sum(out_pre * hidden_post, 2)
            # Softmaxing
            post_soft = F.softmax(pre_soft, dim=0)
            # Applying softmax to corresponding hidden pre-RNN states
            attention = out_pre.permute(2, 0, 1) * post_soft
            # Summing softmax-scaled pre-RNN hidden states and transposing
            attention = torch.sum(attention, 1).t()
            if self.recurrent_type == 'rnn':
                hidden_post = self.recurrent_cell_post(
                    attention, hidden_post
                )
            elif self.recurrent_type == 'lstm':
                hidden_post, state_post = self.recurrent_cell_post(
                    attention, (hidden_post, state_post)
                )

        return self.fc(hidden_post)
