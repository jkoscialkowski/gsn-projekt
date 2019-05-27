# TODO: test if works

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModel(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs,
                 n_layers=1, dropout=0.1): # remember about cuda_device
        """
        Class AttentionModel - implementation of simple Attention architecture

        :param batch_size: size of the batch
        :param n_steps: number of days
        :param n_inputs: number of inputs in the specific day
        :param n_neurons: number of neurons in the
        :param n_outputs: number of output values from the fully connected layer
        :param n_layers: number of layers of the RNN
        :param dropout: dropout probability in the LSTM layer
        """
        super().__init__()

        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn_pre = nn.RNN(input_size=self.n_inputs,
                              hidden_size=self.n_neurons,
                              num_layers=self.n_layers,
                              dropout=self.dropout)
        self.rnn_cell_post = nn.RNNCell(input_size=self.n_neurons,
                                        hidden_size=self.n_neurons)
        self.fc = nn.Linear(self.n_neurons, self.n_outputs)

    def init_hidden(self):
        return torch.zeros(self.n_layers, self.batch_size, self.n_neurons)

    def forward(self, X):
        # remember about transforming data in the right dimension - (n_steps X batch_size X n_inputs)
        # Initialize first hidden state for the pre-RNN with zeros
        hidden = self.init_hidden()
        out_pre, _ = self.rnn_pre(X, hidden)

        # Initialize input to post-RNN with zeros
        hidden_post = torch.zeros(self.batch_size, self.n_neurons)

        for el in range(self.n_neurons):
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

