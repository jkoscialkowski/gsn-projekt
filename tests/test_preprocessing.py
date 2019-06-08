'''
Pytania:
1. Czy processing ma być przeprowadzony na torch
2. Rozumiem, że można dodawać funkcję w reader.py i ewentualnie jakieś dodatkowe funkcję w preprocessing.py
3.

Błędy reader.py:
1. Dlaczego nie wszystkie dataframe są brane pod uwage? - ok, to wynika z dat
2. Dlaczego nie wybieramy tylko tych dat, które nam odpowiadaja?

Fazy preprocessingu:
1. dodawania pustych danych jako średnie
2.

'''

'''
print(tts['train_obs'].size())
print(tts['test_obs'].size())
print(tts['train_y'].size())
print(tts['test_y'].size())
'''

#a = np.array([1,2,3])
#print(a[1:2])
#print(a.torch["AMERICA"].size())
#a.torch['AMERICA'].size()
#format_obs_1 = a.torch['AMERICA'][1, :, :].resize(1, a.torch['AMERICA'][1, :, :].numel())
#format_obs_1 = torch.cat((format_obs_1, a.torch['AMERICA'][1, :, :].resize(1, a.torch['AMERICA'][1, :, :].numel())))
#format_obs_1.size()
'''
a.torch['AMERICA'][11, 1, 1]
a.torch['AMERICA'].size()[0]
len(a.torch['AMERICA'][1, :, 4])

a.torch['AMERICA'][:, 1, :][a.torch['AMERICA'][:, 1, :] != a.torch['AMERICA'][:, 1, :]] =
len(a.torch['AMERICA'][torch.isnan(a.torch['AMERICA'][:, :, :])])
torch.cuda.is_available()
b = AmphibianPreprocess('./data/all_values/stocks/Lodging',
                    datetime.datetime(2010, 1, 4),
                    datetime.datetime(2018, 12, 31))
a.torch['AMERICA'].size()

b.fill_nan()
torch.isnan(b.torch['AMERICA'][:, 1, 1]).sum()
'''
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
from numpy import nanmean, nanstd

'''disable warnings'''
warnings.filterwarnings('ignore')

'''
torch.cuda.get_device_name(0)
torch.cuda.is_available()
'''

a = AmphibianReader('./data/all_values/stocks/Lodging',
                    datetime.datetime(2012, 1, 10),
                    datetime.datetime(2018, 12, 30))
_ = a.read_csvs()
_ = a.get_unique_dates()
_ = a.create_torch()
tts = TrainTestSplit(a, int_start=5, int_end=500)

batch_size = 100
n_neurons = 5
n_outputs = 3
n_layers = 1
n_steps = 4
ds = TimeSeriesDataset(tt_split=tts,
                       int_len=n_steps,
                       transform=transforms.Compose([FillNaN(),
                                                     DummyFillNaN(),
                                                     Normalizing(),
                                                     Formatting(),
                                                     FormattingY()]))
ds.train_obs
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
for i_batch, sample_batched in enumerate(dataloader):
    if i_batch == 0:
        #print(i_batch, torch.cat((sample_batched['observations'][0, 0, 0, :], sample_batched['observations'][0, 0, 1, :]), dim=0))
        model_soft = SoftmaxRegressionModel(batch_size=batch_size, input_size=sample_batched['train_obs'].size(2) * n_steps, n_outputs=n_outputs)
        model = RNNModel(batch_size, n_steps, sample_batched['train_obs'].size(2), n_neurons, n_outputs)
        model_LSTM = LSTMModel(batch_size, n_steps, sample_batched['train_obs'].size(2), n_neurons, n_outputs,
                 num_layers=1, dropout=0.1)
        model_Attention = AttentionModel(batch_size, n_steps, sample_batched['train_obs'].size(2), n_neurons, n_outputs,
                 num_layers=1, dropout=0.1, recurrent_type='rnn', alignment='dotprod')

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