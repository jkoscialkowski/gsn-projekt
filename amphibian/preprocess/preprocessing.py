import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import transforms

class TimeSeriesDataset(Dataset):
    def __init__(self, amReader, int_len = 5, input_reg = 'ASIA_PACIFIC', pred_reg = 'EMEIA', transform=None):
        self.transform = transform
        self.amReader = amReader
        self.whole_set = {'observations': self.amReader.torch[input_reg],
                          'y': self.amReader.torch[pred_reg][:, 5, 0]}
        if self.transform:
            self.whole_set = self.transform(self.whole_set)

        self.len = self.amReader.torch[input_reg].size()[0] - int_len
        self.observations = {}
        self.y = {}
        for i in range(self.len):
            self.observations[i] = self.whole_set['observations'][i:i + int_len, :, :]
            self.y[i] = self.whole_set['y'][i + int_len - 1] # we want to predict Adj Close price

    def __len__(self): # return the length of the data
        return self.len

    def __getitem__(self, item): # return one item on the index
        obs = self.observations[item]
        y = self.y[item]
        sample = {'observations': obs, 'y': y}

        return sample

class Fill_NaN(object):
    def __call__(self, whole_set, value = 1):
        obs, y = whole_set['observations'], whole_set['y']
        for i in range(len(obs[:, 0, 0])):
            if (torch.isnan(y[i])) and (i > 0):
                y[i] = y [i - 1]
            for j in range(len(obs[0, :, 0])):
                for k in range(len(obs[0, 0, :])):
                    if (torch.isnan(obs[i, j, k])) and (i > 0):
                        obs[i, j, k] = obs[i - 1, j, k]
        return {'observations': obs, 'y': y}

class Scaling(object):
    def __call__(self, whole_set):
        obs, y = whole_set['observations'], whole_set['y']
        y_mean = torch.mean(y)
        y_std = torch.std(y)
        for i in range(len(obs[:, 0, 0])):
            y[i] = (y[i] - y_mean)/y_std
        for i in range(len(obs[0, :, 0])):
            for j in range(len(obs[0, 0, :])):
                obs_mean = torch.mean(obs[:, i, j])
                obs_std = torch.std(obs[:, i, j])
                for k in range(len(obs[:, 0, 0])):
                    obs[k, i, j] = (obs[k, i, j] - obs_mean)/obs_std
        return {'observations': obs, 'y': y}

class Train_Test_split(object):

    def __call__(self, sample):
        return 0


'''TO DO:
1. Compose transforms - filling nan -> scalling -> train_test split
2. compute_returns
'''

'''
############## DRAFT, NOTES

class AmphibianPreprocess(AmphibianReader):
    def __init__(self, data_path: str,
                 min_date: datetime.datetime,
                 max_date: datetime.datetime):
        """Class for reading torch tensors with financial time series, which will
        clean them and transform into suitable form for training

        :param data_path: Path to folder with region folders containing quotes
        :param min_date: Minimum date to be read
        :param max_date: Maximum date to be read
        """
        AmphibianReader.__init__(self, data_path, min_date, max_date)
        self.torch = self.create_torch()

    def fill_nan(self, method = 'day_before'):
        """Filling nan in self.torch. Default method is 'day_before'

        :return: self.torch without nans"""

        if method == 'day_before':
            # Iterate over regions
            for reg in self.regions:
                for j in range(len(self.torch[reg][0, :, 0])):
                    for k in range(len(self.torch[reg][0, 0, :])):
                        for i in range(len(self.torch[reg][:, 0, 0])):
                            if torch.isnan(self.torch[reg][i, j, k]):
                                self.torch[reg][i, j, k] = self.torch[reg][i - 1, j, k]
        return self.torch

    # function, which computes returns
    def compute_returns(self, method = 'returns'):
        """Creating tensor with return

        :return: self.torch_returns - tensor with returns"""

        self.fill_nan()
        if method == 'returns':


        return 0

'''

