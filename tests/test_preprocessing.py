import numpy as np
from amphibian.fetch.reader import AmphibianReader
from amphibian.preprocess.train_test_split import TrainTestSplit
from amphibian.preprocess.preprocessing import TimeSeriesDataset, FillNaN, Normalizing, DummyFillNaN, Formatting, FormattingY
import datetime
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from amphibian.architectures import SoftmaxRegressionModel, RNNModel, LSTMModel, AttentionModel
import warnings
from numpy import nanmean, nanstd, floor
import torch.utils.data as data_utils

'''disable warnings'''
warnings.filterwarnings('ignore')

'''
torch.cuda.get_device_name(0)
torch.cuda.is_available()
'''

a = AmphibianReader('./data/all_values/stocks/Lodging',
                    datetime.datetime(2012, 1, 10),
                    datetime.datetime(2018, 1, 30))
_ = a.read_csvs()
_ = a.get_unique_dates()
_ = a.create_torch()
a.torch['EMEIA'].size()
tts = TrainTestSplit(a, int_start=0, int_end=500, train_size=0.8)
tts.whole_set['train_y'].size()
batch_size = 10
n_neurons = 5
n_outputs = 3
n_layers = 1
n_steps = 4
ds = TimeSeriesDataset(tt_split=tts,
                       int_len=n_steps,
                       transform=transforms.Compose([FillNaN(),
                                                     Normalizing(),
                                                     DummyFillNaN(),
                                                     Formatting(),
                                                     FormattingY()]))

'''
for item in range(90):
    print(item % ds.len_train + ds.int_len - 1, int(floor(item / ds.len_train)))
ds[89]
'''

dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
for i_batch, sample_batched in enumerate(dataloader):
    if i_batch == 0:
        #print(i_batch, torch.cat((sample_batched['observations'][0, 0, 0, :], sample_batched['observations'][0, 0, 1, :]), dim=0))
        model_soft = SoftmaxRegressionModel(batch_size=batch_size, input_size=sample_batched['train_obs'].size(2) * n_steps, n_outputs=n_outputs)
        model = RNNModel(batch_size, n_steps, sample_batched['train_obs'].size(2), n_neurons, n_outputs)
        model_LSTM = LSTMModel(batch_size, n_steps, sample_batched['train_obs'].size(2), n_neurons, n_outputs,
                 num_layers=1, dropout=0.1)
        model_Attention = AttentionModel(batch_size, n_steps, sample_batched['train_obs'].size(2), n_neurons, n_outputs,
                 num_layers=1, dropout=0.1, recurrent_type='lstm', alignment='dotprod')

        print('softmax')
        print(sample_batched['train_obs'].mean(dim=0).size())
        print(model_soft(sample_batched['train_obs']))
        print("RNN")
        print(model(sample_batched['train_obs'].permute(1, 0, 2)))
        print("LSTM")
        print(model_LSTM(sample_batched['train_obs'].permute(1, 0, 2)))
        print("Attention")
        print(model_Attention(sample_batched['train_obs'].permute(1, 0, 2)))
        print(sample_batched['train_y'])


"""
Obecny ksztalt: 
.resize(1, 1440)
squeeze() - do zmiejszania wymiarów
"""

'''
X = torch.Tensor([[1., 2.],
                 [2., 1.]])
X.size()


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

output = Variable(torch.FloatTensor([0, 0, 1, 0])).view(1, -1)
target = Variable(torch.LongTensor([3]))

criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
print(output)
print(F.softmax(output))
print(target)
print(-torch.log(F.softmax(output).squeeze()))
print(loss)
'''
'''
w powyzszego wynika, ze 
'''


linear = torch.nn.Linear(10, 3)

input_size = 4
hidden_size = 10
num_layers = 2
batch_size = 3
seq_len = 2
n_outputs = 3

#input.view(7, 10, 5).view(7, 50).size()

rnn = torch.nn.LSTM(input_size, hidden_size, num_layers)
input = torch.randn(seq_len, batch_size, input_size)
h0 = torch.randn(num_layers, batch_size, hidden_size)
c0 = torch.randn(num_layers, batch_size, hidden_size)
output, (hn, cn) = rnn(input, (h0, c0))
input = input.view(batch_size, seq_len * input_size)
hn.size()
fc = torch.nn.Linear(in_features=input_size * seq_len, out_features=n_outputs)
fc(input)

srednia = input.mean(dim=0)
odchylenie = input.std(dim=0)
jedynki = torch.ones(2, 3, 4)
jedynki.resize(1,1,24)
input.view(3, 2, 4)
input.permute(1, 0, 2).resize(3, 2 * 4)
input
(jedynki - srednia) / odchylenie
srednia[srednia < 0] = 0

torch.Tensor.diff(srednia)
test = (srednia[1:] - srednia[:-1]) / srednia[:-1]
test[srednia[1:] > 0] = 0
input.resize(2, 1, input[0,:,:].numel()).size()

srednia[srednia < 0] = 0
srednia[(srednia == 0) + 1] = srednia[:-1]
srednia.fillna(srednia[1:])

(srednia == 0)[1:]
srednia[srednia == 0] = srednia[:-1][(srednia == 0)[1:]]

input[input < 0].mean(dim=0)
input[1:, :][input[1:, :] == 0] = input[:-1, :][(input == 0)[1:, :]]
torch.nanmean(srednia)

torch.tensor(nanmean(input.numpy(), axis=0))

test = 'test3'
if test in ['test', 'test2']:
    print(1)



test = torch.zeros(5)
test