import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Simple_RNN(nn.Module):
    def __init__(self, n_inputs, n_neurons):
        super(Simple_RNN, self).__init__()

        self.Wx = torch.randn(n_inputs, n_neurons)
        self.Wy = torch.randn(n_neurons, n_neurons)

        self.b = torch.zeros(1, n_neurons)

    def forward(self, X0, X1):
        self.Y0 = torch.tanh(torch.mm(X0, self.Wx) + self.b)

        self.Y1 = torch.tanh(torch.mm(self.Y0, self.Wy) + torch.mm(X1, self.Wx) + self.b)

        return self.Y0, self.Y1

