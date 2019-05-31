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
from amphibian.preprocess.preprocessing import TimeSeriesDataset, Fill_NaN, Normalizing, Dummy_Fill_NaN, Formatting
import datetime
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from amphibian.architectures import RNN_model, LSTMModel, AttentionModel

a = AmphibianReader('./data/all_values/stocks/Lodging',
                    datetime.datetime(2012, 1, 10),
                    datetime.datetime(2012, 1, 30))
_ = a.read_csvs()
_ = a.get_unique_dates()
_ = a.create_torch()
a.torch["AMERICA"].size()
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

n_steps = 5
batch_size = 5
n_neurons = 5
n_outputs = 1
n_layers = 1
ds = TimeSeriesDataset(amReader=a, int_len = n_steps, transform=transforms.Compose([Fill_NaN(), Dummy_Fill_NaN(), Normalizing(), Formatting()]))
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
for i_batch, sample_batched in enumerate(dataloader):
    if i_batch == 0:
        #print(i_batch, torch.cat((sample_batched['observations'][0, 0, 0, :], sample_batched['observations'][0, 0, 1, :]), dim=0))
        model = RNN_model(batch_size, n_steps, sample_batched['observations'].size(2), n_neurons, n_outputs, n_layers)
        model_LSTM = LSTMModel(batch_size, n_steps, sample_batched['observations'].size(2), n_neurons, n_outputs,
                 num_layers=1, dropout=0.1)
        model_Attention = AttentionModel(batch_size, n_steps, sample_batched['observations'].size(2), n_neurons, n_outputs,
                 num_layers=1, dropout=0.1, recurrent_type='rnn',
                 alignment='dotprod')
        print(sample_batched['observations'])
        print("RNN")
        print(model(sample_batched['observations'].permute(1, 0, 2)))
        print("LSTM")
        print(model_LSTM(sample_batched['observations'].permute(1, 0, 2)))
        print("Attention")
        print(model_Attention(sample_batched['observations'].permute(1, 0, 2)))
        #print(torch.ones((2, ), dtype=torch.int8))


"""
Obecny ksztalt: 
.resize(1, 1440)
squeeze() - do zmiejszania wymiarów
"""
