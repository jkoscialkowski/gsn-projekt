import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set CUDA if available
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


class SoftmaxRegressionModel(nn.Module):
    def __init__(self, batch_size: int, seq_len: int, input_size: int,
                 n_outputs: int):
        """Class SoftmaxRegressionModel - implementation of Softmax Regression.

        :param batch_size: size of the batch
        :param seq_len: only for compatibility with training classes
        :param input_size: size of input
        :param n_outputs: size of output
        """
        super().__init__()
        # Setting parameters
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.fc = nn.Linear(in_features=self.input_size * self.seq_len,
                            out_features=self.n_outputs)

    def forward(self, X):
        flattened = X.contiguous().view(-1, self.input_size * self.seq_len)
        out = self.fc(flattened)
        return out


class RNNModel(nn.Module):
    def __init__(self, batch_size: int, seq_len: int, input_size: int,
                 hidden_size: int, n_outputs: int, num_layers: int = 1,
                 dropout: float = 0.1):
        """Class RNNModel - implementation of simple RNN model with a dense
        layer just before output.

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

    def init_hidden(self, bs):
        return torch.zeros(self.num_layers, bs, self.hidden_size,
                           requires_grad=False, device=DEVICE)

    def forward(self, X):
        hidden = self.init_hidden(X.shape[1])
        out, _ = self.rnn(X, hidden)
        out = self.fc(out[-1, :, :].squeeze())
        return out


class GRUModel(nn.Module):
    def __init__(self, batch_size: int, seq_len: int, input_size: int,
                 hidden_size: int, n_outputs: int, num_layers: int = 1,
                 dropout: float = 0.1):
        """
        Class GRUModel - implementation of the Gated Recurrent Unit with a dense
        layer just before output.

        :param batch_size: size of the batch
        :param seq_len: number of days
        :param input_size: number of inputs in the specific day
        :param hidden_size: number of features in the hidden state
        :param n_outputs: number of output values from the fully connected layer
        :param num_layers: number of layers of the GRU
        :param dropout: dropout
        """
        super().__init__()
        # Setting parameters
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.n_outputs)

    def init_hidden(self, bs):
        return torch.zeros(self.num_layers, bs, self.hidden_size,
                           requires_grad=False, device=DEVICE)

    def forward(self, X):
        hidden = self.init_hidden(X.shape[1])
        out, _ = self.gru(X, hidden)

        out = self.fc(out[-1, :, :].squeeze())
        return out


class LSTMModel(nn.Module):
    def __init__(self, batch_size: int, seq_len: int, input_size: int,
                 hidden_size: int, n_outputs: int, num_layers: int = 1,
                 dropout: float = 0.1):
        """
        Class LSTMModel - implementation of Long Short-Term Memory with a
        single dense layer just before output.

        :param batch_size: size of the batch
        :param seq_len: number of days
        :param input_size: number of inputs in the specific day
        :param hidden_size: number of features in the hidden state
        :param n_outputs: number of output values from the fully connected layer
        :param num_layers: number of layers of the LSTM
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

    def init_hidden(self, bs):
        return (torch.zeros(self.num_layers, bs, self.hidden_size,
                            requires_grad=False, device=DEVICE),
                torch.zeros(self.num_layers, bs, self.hidden_size,
                            requires_grad=False, device=DEVICE))

    def forward(self, X):
        hidden = self.init_hidden(X.shape[1])
        out, _ = self.lstm(X, hidden)
        out = self.fc(out[-1, :, :].squeeze())
        return out


class AttentionModel(nn.Module):
    def __init__(self, batch_size: int, seq_len: int, input_size: int,
                 hidden_size: int, n_outputs: int, num_layers: int = 1,
                 dropout: float = 0.1, recurrent_type: str = 'rnn',
                 alignment: str = 'dotprod', additional_y_layer: str = 'no',
                 switch_cells: str = 'no'):
        """
        Class AttentionModel - implementation of Attention Mechanism, with the
        possible of extensions: additional 'state' layer; additional 'switch state' layer

        :param batch_size: size of the batch
        :param seq_len: number of days
        :param input_size: number of inputs in the specific day
        :param hidden_size: number of neurons in the
        :param n_outputs: number of output values from the fully connected layer
        :param num_layers: number of layers in the in the pre-Attention
        recurrent layer
        :param dropout: dropout probability in the pre-Attention recurrent layer
        :param recurrent_type: whether to use RNN, GRU or LSTM layers
        :param alignment: whether to use dot product or feedforward NN to align
        hidden states of pre- and post-Attention layers
        :param additional_y_layer: whether Attention model should be extended with 'state' layer
        :param switch_cells: whether Attention model with 'state' layer should be extended with switch cells layer
        """
        super().__init__()
        assert recurrent_type in ['rnn', 'lstm', 'gru']
        assert alignment in ['dotprod', 'ffnn']
        assert additional_y_layer in ['no', 'yes']
        assert switch_cells in ['no', 'yes']
        # Setting parameters
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        # Defining alignment behaviour
        if self.alignment == 'ffnn':
            # Xavier initialisation
            self.w_a = nn.Parameter(
                torch.randn(hidden_size, hidden_size,
                            requires_grad=True) / np.sqrt(hidden_size),
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
            if switch_cells == 'yes':
                self.recurrent_cell_post_1 = nn.RNNCell(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size)
                self.recurrent_cell_post_2 = nn.RNNCell(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size)
        elif recurrent_type == 'lstm':
            self.recurrent_pre = nn.LSTM(input_size=self.input_size,
                                         hidden_size=self.hidden_size,
                                         num_layers=self.num_layers,
                                         dropout=self.dropout)
            self.recurrent_cell_post = nn.LSTMCell(input_size=self.hidden_size,
                                                   hidden_size=self.hidden_size)
            if switch_cells == 'yes':
                self.recurrent_cell_post_1 = nn.LSTMCell(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size)
                self.recurrent_cell_post_2 = nn.LSTMCell(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size)
        elif recurrent_type == 'gru':
            self.recurrent_pre = nn.GRU(input_size=self.input_size,
                                        hidden_size=self.hidden_size,
                                        num_layers=self.num_layers,
                                        dropout=self.dropout)
            self.recurrent_cell_post = nn.GRUCell(input_size=self.hidden_size,
                                                  hidden_size=self.hidden_size)
            if switch_cells == 'yes':
                self.recurrent_cell_post_1 = nn.GRUCell(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size)
                self.recurrent_cell_post_2 = nn.GRUCell(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size)

        if self.additional_y_layer == 'yes':
            self.add_y_layer = nn.Linear(self.hidden_size + 1, self.hidden_size)

        self.fc = nn.Linear(self.hidden_size, self.n_outputs)

    def init_hidden(self, which, bs):
        """Hidden state initialisation.

        :param which: whether to initialise for pre- or post-Attention layer
        :param bs: batch_size
        :return: initialised hidden state
        """
        assert which in ['pre', 'post']
        if which == 'pre':
            dims = self.num_layers, bs, self.hidden_size
        elif which == 'post':
            dims = bs, self.hidden_size
        return torch.zeros(*dims, requires_grad=False, device=DEVICE)

    def forward(self, X, y=None):
        # Initialize first hidden state for the pre-RNN with zeros
        if self.recurrent_type in ['rnn', 'gru']:
            hidden_pre = self.init_hidden('pre', X.shape[1])
            out_pre, _ = self.recurrent_pre(X, hidden_pre)
        elif self.recurrent_type == 'lstm':
            hidden_pre, state_pre = (self.init_hidden('pre', X.shape[1]),
                                     self.init_hidden('pre', X.shape[1]))
            out_pre, _ = self.recurrent_pre(X, (hidden_pre, state_pre))

        # Initialize input to post-RNN with zeros
        hidden_post = self.init_hidden('post', X.shape[1])
        if self.recurrent_type == 'lstm':
            state_post = self.init_hidden('post', X.shape[1])
        for el in range(self.seq_len):
            if self.additional_y_layer == 'yes':
                if el > 0:
                    hidden_post = self.add_y_layer(
                        torch.cat((hidden_post, y[el - 1, :].unsqueeze(0).permute(1, 0)), dim=1))
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
            if self.switch_cells == 'yes':
                if el < self.seq_len - 1:
                    for i in range(y[el, :].size(0)):
                        if y[el, i] == 0:
                            if self.recurrent_type in ['rnn', 'gru']:
                                hidden_post = self.recurrent_cell_post(
                                    attention, hidden_post
                                )
                            elif self.recurrent_type == 'lstm':
                                hidden_post, state_post = self.recurrent_cell_post(
                                    attention, (hidden_post, state_post)
                                )
                        elif y[el, i] == 1:
                            if self.recurrent_type in ['rnn', 'gru']:
                                hidden_post = self.recurrent_cell_post_1(
                                    attention, hidden_post
                                )
                            elif self.recurrent_type == 'lstm':
                                hidden_post, state_post = self.recurrent_cell_post_1(
                                    attention, (hidden_post, state_post)
                                )
                        elif y[el, i] == 2:
                            if self.recurrent_type in ['rnn', 'gru']:
                                hidden_post = self.recurrent_cell_post_2(
                                    attention, hidden_post
                                )
                            elif self.recurrent_type == 'lstm':
                                hidden_post, state_post = self.recurrent_cell_post_2(
                                    attention, (hidden_post, state_post)
                                )
                else:
                    if self.recurrent_type in ['rnn', 'gru']:
                        hidden_post = self.recurrent_cell_post_1(
                            attention, hidden_post
                        )
                    elif self.recurrent_type == 'lstm':
                        hidden_post, state_post = self.recurrent_cell_post_1(
                            attention, (hidden_post, state_post)
                        )
            else:
                if self.recurrent_type in ['rnn', 'gru']:
                    hidden_post = self.recurrent_cell_post(
                        attention, hidden_post
                    )
                elif self.recurrent_type == 'lstm':
                    hidden_post, state_post = self.recurrent_cell_post(
                        attention, (hidden_post, state_post)
                    )
        return self.fc(hidden_post)
