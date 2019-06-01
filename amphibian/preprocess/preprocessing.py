"""
TODO 1: Upgrade Fill_NaN - with averages
"""

"""import modules"""
import torch
from torch.utils.data import Dataset

"""classes"""
class TimeSeriesDataset(Dataset):
    def __init__(self, ttSplit, int_len = 5, transform=None):
        """
        Class TimeSeriesDataset - object of this class is the

        :param ttSpplit: object from Train_test_split class
        :param int_len: number of days in one observation
        :param input_reg: input region
        :param pred_reg: region, which we would like to predict
        :param transform: sequence of transformations, default: None
        """
        self.transform = transform
        self.whole_set = ttSplit

        if self.transform:
            self.whole_set = self.transform(self.whole_set)

        self.len_train = self.whole_set['train_y'].size(0) - int_len + 1
        self.len_test = self.whole_set['test_y'].size(0) - int_len + 1
        self.train_obs = {}
        self.train_y = {}
        for i in range(self.len_train):
            self.train_obs[i] = self.whole_set['train_obs'][i:i + int_len, :]
            self.train_y[i] = self.whole_set['train_y'][i + int_len - 1] # we want to predict Adj Close price
        self.test_obs = {}
        self.test_y = {}
        for i in range(self.len_test):
            self.test_obs[i] = self.whole_set['test_obs'][i:i + int_len, :]
            self.test_y[i] = self.whole_set['test_y'][i + int_len - 1]  # we want to predict Adj Close price

    def __len__(self):
        """
        :return: length of data
        """
        return self.len_train

    def __getitem__(self, item1):
        """
        :param item: index
        :return: one item on the given index
        """
        obs = self.train_obs[item1]
        y = self.train_y[item1]
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
        for i in range(train_obs.size(0)):
            if (torch.isnan(train_y[i])) and (i > 0):
                train_y[i] = train_y[i - 1]
            for j in range(train_obs.size(1)):
                for k in range(train_obs.size(2)):
                    if (torch.isnan(train_obs[i, j, k])) and (i > 0):
                        train_obs[i, j, k] = train_obs[i - 1, j, k]
        # first observations in test set
        if torch.isnan(test_y[0]):
            test_y[0] = train_y[-1]
        for j in range(test_obs.size(1)):
            for k in range(test_obs.size(2)):
                if torch.isnan(test_obs[0, j, k]):
                    test_obs[0, j, k] = train_obs[-1, j, k]
        # Test set
        for i in range(1, test_obs.size(0)):
            if torch.isnan(test_y[i]):
                test_y[i] = test_y[i - 1]
            for j in range(test_obs.size(1)):
                for k in range(test_obs.size(2)):
                    if torch.isnan(train_obs[i, j, k]):
                        train_obs[i, j, k] = train_obs[i - 1, j, k]

        return {'train_obs': train_obs, 'train_y': train_y,
                'test_obs': test_obs, 'test_y': test_y}

class Dummy_Fill_NaN(object):
    def __call__(self, whole_set):
        """
        Dummy NaN filling
        :param whole_set: set of observations
        :return: set of observations with NaN filled
        """
        train_obs, train_y = whole_set['train_obs'], whole_set['train_y']
        test_obs, test_y = whole_set['test_obs'], whole_set['test_y']
        # training set
        for i in range(train_obs.size(0)):
            if torch.isnan(train_y[i]):
                train_y[i] = 0
            for j in range(train_obs.size(1)):
                for k in range(train_obs.size(2)):
                    if torch.isnan(train_obs[i, j, k]):
                        train_obs[i, j, k] = 0
        # test set
        for i in range(test_obs.size(0)):
            if torch.isnan(test_y[i]):
                test_y[i] = 0
            for j in range(test_obs.size(1)):
                for k in range(test_obs.size(2)):
                    if torch.isnan(test_obs[i, j, k]):
                        test_obs[i, j, k] = 0

        return {'train_obs': train_obs, 'train_y': train_y,
                'test_obs': test_obs, 'test_y': test_y}

class Normalizing(object):
    def __call__(self, whole_set, eps = 10 ** -3):
        """
        :param whole_set: set of observarions
        :return: normalized set of observations
        """
        train_obs, train_y = whole_set['train_obs'], whole_set['train_y']
        test_obs, test_y = whole_set['test_obs'], whole_set['test_y']
        for i in range(train_obs.size(1)):
            for j in range(train_obs.size(2)):
                train_obs_mean = torch.mean(train_obs[:, i, j][train_obs[:, i, j] == train_obs[:, i, j]])
                train_obs_std = torch.std(train_obs[:, i, j][train_obs[:, i, j] == train_obs[:, i, j]])
                if train_obs_std < eps:
                    train_obs_std = eps
                for k in range(train_obs.size(0)):
                    train_obs[k, i, j] = (train_obs[k, i, j] - train_obs_mean)/train_obs_std
                for k in range(test_obs.size(0)):
                    test_obs[k, i, j] = (train_obs[k, i, j] - train_obs_mean)/train_obs_std

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
        # training set
        format_train_obs = train_obs[0, :, :].resize(1, train_obs[0, :, :].numel())
        for i in range(1, train_obs.size(0)):
            format_train_obs = torch.cat((format_train_obs,
                                          train_obs[i, :, :].resize(1, train_obs[i, :, :].numel())))
        # test set
        format_test_obs = test_obs[0, :, :].resize(1, test_obs[0, :, :].numel())
        for i in range(1, test_obs.size(0)):
            format_test_obs = torch.cat((format_test_obs,
                                         test_obs[i, :, :].resize(1, test_obs[i, :, :].numel())))

        return {'train_obs': format_train_obs, 'train_y': train_y,
                'test_obs': format_test_obs, 'test_y': test_y}

class Formatting_y(object):
    def __call__(self, whole_set, eps_up = 0.01, eps_down = -0.01):
        """
        :param whole_set: set of observations
        :param eps_up: threshold for return being regarded as 'up'
        :param eps_down: threshold for return being regarded as 'down'
        :return: transformed y observarions into three states: 1 - up; 0 - stable; -1 - down
        """
        train_obs, train_y = whole_set['train_obs'], whole_set['train_y']
        test_obs, test_y = whole_set['test_obs'], whole_set['test_y']
        # training set
        format_train_y = torch.empty(len(train_y))
        format_train_y[0] = 0
        for i in range(1, len(train_y)):
            if (train_y[i] - train_y[i - 1]) / train_y[i - 1] > eps_up:
                format_train_y[i] = 2
            elif (train_y[i] - train_y[i - 1]) / train_y[i - 1] < eps_down:
                format_train_y[i] = 0
            else:
                format_train_y[i] = 1
        # test set
        format_test_y = torch.empty(len(test_y))
        format_test_y[0] = 0
        for i in range(1, len(test_y)):
            if (test_y[i] - test_y[i - 1]) / test_y[i - 1] > eps_up:
                format_test_y[i] = 2
            elif (test_y[i] - test_y[i - 1]) / test_y[i - 1] < eps_down:
                format_test_y[i] = 0
            else:
                format_test_y[i] = 1

        return {'train_obs': train_obs, 'train_y': format_train_y,
                'test_obs': test_obs, 'test_y': format_test_y}

