"""
TODO 1: Upgrade Fill_NaN - with averages
"""

"""import modules"""
import torch
from torch.utils.data import Dataset
from numpy import nanmean, nanstd, floor

"""classes"""
class TimeSeriesDataset(Dataset):
    def __init__(self, ttSplit, int_len=5, transform=None):
        """
        Class TimeSeriesDataset

        :param ttSpplit: object from Train_test_split class
        :param int_len: number of days in one observation
        :param input_reg: input region
        :param pred_reg: region, which we would like to predict
        :param transform: sequence of transformations, default: None
        """
        self.transform = transform
        self.whole_set = ttSplit
        self.int_len = int_len

        if self.transform:
            self.whole_set = self.transform(self.whole_set)

        self.len_train = self.whole_set['train_y'].size(0) - self.int_len + 1
        self.len_test = self.whole_set['test_y'].size(0) - self.int_len + 1

    def __len__(self):
        """
        :return: length of data
        """
        return self.len_train

    def __getitem__(self, item):
        """
        :param item: index
        :return: one item on the given index
        """
        obs = self.whole_set['train_obs'][item % self.len_train:item % self.len_train + self.int_len, :]
        y = self.whole_set['train_y'][item % self.len_train + self.int_len - 1, int(floor(item / self.len_train))]
        sample = {'train_obs': obs, 'train_y': y}

        return sample

class Fill_NaN(object):
    def __call__(self, whole_set):
        """
        :param whole_set: set of observations
        :return: set of NaN values filled with prior observation
        """
        train_obs, train_y = whole_set['train_obs'], whole_set['train_y']
        test_obs, test_y = whole_set['test_obs'], whole_set['test_y']

        # Training set
        train_y[1:, :][torch.isnan(train_y[1:, :])] = train_y[:-1, :][torch.isnan(train_y)[1:, :]]
        train_obs[1:, :, :][torch.isnan(train_obs[1:, :, :])] = train_obs[:-1, :, :][torch.isnan(train_obs)[1:, :, :]]
        # first observations in test set
        test_y[0, :][torch.isnan(test_y[0, :])] = train_y[-1, :][torch.isnan(test_y[0, :])]
        test_obs[0, :, :][torch.isnan(test_obs[0, :, :])] = train_obs[-1, :, :][torch.isnan(test_obs[0, :, :])]
        # Test set
        test_y[1:, :][torch.isnan(test_y[1:, :])] = test_y[:-1, :][torch.isnan(test_y)[1:, :]]
        test_obs[1:, :, :][torch.isnan(test_obs[1:, :, :])] = test_obs[:-1, :, :][torch.isnan(test_obs)[1:, :, :]]

        return {'train_obs': train_obs, 'train_y': train_y,
                'test_obs': test_obs, 'test_y': test_y}

class Normalizing(object):
    def __call__(self, whole_set, eps=10 ** -3):
        """
        :param whole_set: set of observarions
        :return: normalized set of observations
        """
        train_obs, train_y = whole_set['train_obs'], whole_set['train_y']
        test_obs, test_y = whole_set['test_obs'], whole_set['test_y']

        train_obs_mean = torch.tensor(nanmean(train_obs.numpy(), axis=0))
        train_obs_std = torch.tensor(nanstd(train_obs.numpy(), axis=0))
        train_obs_std[train_obs_std < eps] = eps
        # training set
        train_obs = (train_obs - train_obs_mean)/train_obs_std
        # test set
        test_obs = (test_obs - train_obs_mean)/train_obs_std

        return {'train_obs': train_obs, 'train_y': train_y,
                'test_obs': test_obs, 'test_y': test_y}

class Dummy_Fill_NaN(object):
    def __call__(self, whole_set):
        """
        :param whole_set: set of observations
        :return: set of observations with NaN filled
        """
        train_obs, train_y = whole_set['train_obs'], whole_set['train_y']
        test_obs, test_y = whole_set['test_obs'], whole_set['test_y']

        # training set
        train_y[torch.isnan(train_y)] = 0
        train_obs[torch.isnan(train_obs)] = 0
        # test set
        test_y[torch.isnan(test_y)] = 0
        test_obs[torch.isnan(test_obs)] = 0

        return {'train_obs': train_obs, 'train_y': train_y,
                'test_obs': test_obs, 'test_y': test_y}

class Formatting(object):
    def __call__(self, whole_set):
        """
        :param whole_set: set of observations
        :return: set of observations in the right shape --- suitable for NN (see architectures.py)
        """
        train_obs, train_y = whole_set['train_obs'], whole_set['train_y']
        test_obs, test_y = whole_set['test_obs'], whole_set['test_y']

        format_train_obs = train_obs.resize(train_obs.size(0), 1, train_obs[0, :, :].numel()).squeeze()
        format_test_obs = test_obs.resize(test_obs.size(0), 1, test_obs[0, :, :].numel()).squeeze()

        return {'train_obs': format_train_obs, 'train_y': train_y,
                'test_obs': format_test_obs, 'test_y': test_y}

class Formatting_y(object):
    def __call__(self, whole_set, eps_up=0.005, eps_down=-0.005):
        """
        :param whole_set: set of observations
        :param eps_up: threshold for return being regarded as 'up'
        :param eps_down: threshold for return being regarded as 'down'
        :return: transformed y observarions into three states: 2 - up; 1 - stable; 0 - down
        """
        train_obs, train_y = whole_set['train_obs'], whole_set['train_y']
        test_obs, test_y = whole_set['test_obs'], whole_set['test_y']
        # training set
        format_train_y = torch.ones(train_y.size())
        train_y[1:, :] = (train_y[1:, :] - train_y[:-1, :]) / train_y[:-1, :]
        format_train_y[1:, :][train_y[1:, :] > eps_up] = 2
        format_train_y[1:, :][train_y[1:, :] < eps_down] = 0
        # test set
        format_test_y = torch.ones(test_y.size())
        test_y[1:, :] = (test_y[1:, :] - test_y[:-1, :]) / test_y[:-1, :]
        format_test_y[1:, :][test_y[1:, :] > eps_up] = 2
        format_test_y[1:, :][test_y[1:, :] < eps_down] = 0

        return {'train_obs': train_obs, 'train_y': format_train_y,
                'test_obs': test_obs, 'test_y': format_test_y}

