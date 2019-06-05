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
import numpy as np
from amphibian.fetch.reader import AmphibianReader
from amphibian.preprocess.train_test_split import Train_test_split
from amphibian.preprocess.preprocessing import TimeSeriesDataset, Fill_NaN, Normalizing, Dummy_Fill_NaN, Formatting, Formatting_y
import datetime
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from amphibian.architectures import SoftmaxRegressionModel, RNNModel, LSTMModel, AttentionModel
import warnings

'''disable warnings'''
warnings.filterwarnings('ignore')

'''
torch.cuda.get_device_name(0)
torch.cuda.is_available()
'''

a = AmphibianReader('./data/all_values/stocks/Lodging',
                    datetime.datetime(2012, 1, 10),
                    datetime.datetime(2012, 1, 30))
_ = a.read_csvs()
_ = a.get_unique_dates()
_ = a.create_torch()
tts = Train_test_split(a, int_start=5, int_end=15)
'''
print(tts['train_obs'].size())
print(tts['test_obs'].size())
print(tts['train_y'].size())
print(tts['test_y'].size())
'''
print(tts['train_obs'].size())

ds = TimeSeriesDataset(ttSplit=tts,
                       int_len=3,
                       transform=transforms.Compose([Fill_NaN(),
                                                     Normalizing(),
                                                     Dummy_Fill_NaN(),
                                                     Formatting(),
                                                     Formatting_y()]))

print(ds.train_obs.size())
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


n_steps = 3
batch_size = 2
n_neurons = 5
n_outputs = 3
n_layers = 1

dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
for i_batch, sample_batched in enumerate(dataloader):
    if i_batch == 0:
        #print(i_batch, torch.cat((sample_batched['observations'][0, 0, 0, :], sample_batched['observations'][0, 0, 1, :]), dim=0))
        model_soft = SoftmaxRegressionModel(batch_size, sample_batched['train_obs'].size(2) * n_steps, n_outputs)
        model = RNNModel(batch_size, n_steps, sample_batched['train_obs'].size(2), n_neurons, n_outputs)
        model_LSTM = LSTMModel(batch_size, n_steps, sample_batched['train_obs'].size(2), n_neurons, n_outputs,
                 num_layers=1, dropout=0.1)
        model_Attention = AttentionModel(batch_size, n_steps, sample_batched['train_obs'].size(2), n_neurons, n_outputs,
                 num_layers=1, dropout=0.1, recurrent_type='rnn', alignment='dotprod')

        print('softmax')
        print(model_soft(sample_batched['train_obs']))
        print("RNN")
        print(model(sample_batched['train_obs'].permute(1, 0, 2)))
        print("LSTM")
        print(model_LSTM(sample_batched['train_obs'].permute(1, 0, 2)))
        print("Attention")
        print(model_Attention(sample_batched['train_obs'].permute(1, 0, 2)))


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

input_size = 5
hidden_size = 10
num_layers = 2
batch_size = 7
seq_len = 10
n_outputs = 3

input.view(7, 10, 5).view(7, 50).size()

rnn = torch.nn.LSTM(input_size, hidden_size, num_layers)
input = torch.randn(seq_len, batch_size, input_size)
h0 = torch.randn(num_layers, batch_size, hidden_size)
c0 = torch.randn(num_layers, batch_size, hidden_size)
output, (hn, cn) = rnn(input, (h0, c0))
output.size()
hn.size()
fc = torch.nn.Linear(hidden_size, n_outputs)
fc(output[-1, :, :])
