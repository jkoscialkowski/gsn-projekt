import torch
import torch.nn as nn

class RNN_model(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs, n_layers): # remember about cuda_device
        """
        Class RNN_model - object of this class is the architecture which consists of RNN layers and fully connected layer

        :param batch_size: size of the batch
        :param n_steps: number of days
        :param n_inputs: number of inputs in the specific day
        :param n_neurons: number of neurons in the
        :param n_outputs: number of output values from the fully connected layer
        :param n_layers: number of layers of the RNN
        """
        super(RNN_model, self).__init__()

        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers

        self.rnn = nn.RNN(self.n_inputs, self.n_neurons)
        self.fc = nn.Linear(self.n_neurons, self.n_outputs) # Fully connected layer

    def init_hidden(self, ):
        return torch.zeros(self.n_layers, self.batch_size, self.n_neurons)

    def forward(self, X):
        # remember about transforming data in the right dimension - (n_steps X batch_size X n_inputs)
        self.hidden = self.init_hidden()

        out, self.hidden = self.rnn(X, self.hidden)
        out = self.fc(self.hidden)

        return self.rnn(X, self.hidden), out

# TEST
X_batch = torch.tensor([[[1, 2, 5]],
                        [[1, 2, 3]],
                        [[1, 2, 3]],
                        [[0, 1, 2]]],
                        dtype = torch.float
                       )
batch_size = X_batch.size(1)
n_inputs = X_batch.size(2)
n_steps = X_batch.size(0)

n_neurons = 2
n_outputs = 1
n_layers = 1 # upgrade class for abitrary number of layers

torch.zeros(1, 4, 3)
model = RNN_model(batch_size, n_steps, n_inputs, n_neurons, n_outputs, n_layers)
model(X_batch)