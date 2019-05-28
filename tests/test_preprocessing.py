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
from amphibian.preprocess.preprocessing import TimeSeriesDataset
import datetime
import pandas as pd
from torch.utils.data import DataLoader

a = AmphibianReader('./data/all_values/stocks/Lodging',
                    datetime.datetime(2010, 1, 4),
                    datetime.datetime(2018, 12, 31))
_ = a.read_csvs()
_ = a.get_unique_dates()
_ = a.create_torch()
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

ds = TimeSeriesDataset(a)
dataloader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4, )

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched[0].size())


