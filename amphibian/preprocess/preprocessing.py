"""
TODO 1: Upgrade Fill_NaN - with averages,
TODO 2: Train_test_split
TODO 3: Upgrade normalizing - sometimes std < epsilon or
"""

"""import modules"""
import torch
from torch.utils.data import Dataset

"""classes"""
class TimeSeriesDataset(Dataset):
    def __init__(self, amReader, int_len = 5, input_reg = 'ASIA_PACIFIC', pred_reg = 'EMEIA', transform=None):
        """
        Class TimeSeriesDataset - object of this class is the

        :param amReader: object from AmphibianReader class
        :param int_len: number of days in one observation
        :param input_reg: input region
        :param pred_reg: region, which we would like to predict
        :param transform: sequence of transformations, default: None
        """
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
            self.observations[i] = self.whole_set['observations'][i:i + int_len, :]
            self.y[i] = self.whole_set['y'][i + int_len - 1] # we want to predict Adj Close price

    def __len__(self):
        """
        :return: length of data
        """
        return self.len

    def __getitem__(self, item):
        """
        :param item: index
        :return: one item on the given index
        """
        obs = self.observations[item]
        y = self.y[item]
        sample = {'observations': obs, 'y': y}

        return sample

class Fill_NaN(object):
    def __call__(self, whole_set):
        """
        :param whole_set: set of observations
        :return: set of observations with NaN filled with last observation
        """
        obs, y = whole_set['observations'], whole_set['y']
        for i in range(obs.size(0)):
            if (torch.isnan(y[i])) and (i > 0):
                y[i] = y[i - 1]
            for j in range(obs.size(1)):
                for k in range(obs.size(2)):
                    if (torch.isnan(obs[i, j, k])) and (i > 0):
                        obs[i, j, k] = obs[i - 1, j, k]
        return {'observations': obs, 'y': y}

class Dummy_Fill_NaN(object):
    def __call__(self, whole_set):
        """
        Dummy NaN filling
        :param whole_set: set of observations
        :return: set of observations with NaN filled
        """
        obs, y = whole_set['observations'], whole_set['y']
        for i in range(obs.size(0)):
            if (torch.isnan(y[i])):
                y[i] = 0
            for j in range(obs.size(1)):
                for k in range(obs.size(2)):
                    if (torch.isnan(obs[i, j, k])):
                        obs[i, j, k] = 0
        return {'observations': obs, 'y': y}

class Normalizing(object):
    def __call__(self, whole_set):
        """
        :param whole_set: set of observarions
        :return: normalized set of observations
        """
        obs, y = whole_set['observations'], whole_set['y']
        y_mean = torch.mean(y)
        y_std = torch.std(y)
        if y_std < 10 ** -3:
            y_std == 10 ** -3
        for i in range(obs.size(0)):
            y[i] = (y[i] - y_mean)/y_std
        for i in range(obs.size(1)):
            for j in range(obs.size(2)):
                obs_mean = torch.mean(obs[:, i, j])
                obs_std = torch.std(obs[:, i, j])
                if obs_std < 10 ** -3:
                    obs_std = 10 ** -3
                for k in range(obs.size(0)):
                    obs[k, i, j] = (obs[k, i, j] - obs_mean)/obs_std
        return {'observations': obs, 'y': y}

class Formatting(object):
    def __call__(self, whole_set):
        """
        :param whole_set: set of observations
        :return: set of observations in the right shape --- suitable for NN (see architectures.py)
        """
        obs, y = whole_set['observations'], whole_set['y']
        format_obs = obs[0, :, :].resize(1, obs[0, :, :].numel())
        for i in range(1, obs.size(0)):
            format_obs = torch.cat((format_obs, obs[i, :, :].resize(1, obs[i, :, :].numel())))
        return {'observations': format_obs, 'y': y}

class Train_Test_split(object):

    def __call__(self, sample):
        return 0

