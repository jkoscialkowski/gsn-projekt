import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from livelossplot import PlotLosses
from scipy import stats
from torch.utils.data import DataLoader
from torchvision import transforms

import amphibian.preprocess.preprocessing as preproc
from amphibian.preprocess.train_test_split import TrainTestSplit

# Set CUDA if available
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# Set implemented SingleTrainer parameters which can be passed to CV
IMPLEMENTED_ARCHITECTURES = [
    'SoftmaxRegressionModel', 'RNNModel', 'GRUModel', 'LSTMModel',
    'AttentionModel'
]
NON_MODEL_PARAMETERS = [
    'learning_rate',
    'max_epochs',
    'early_stopping_patience'
]


class SingleTrainer:
    def __init__(self, model, batch_size: int, learning_rate: float = 1e-3,
                 max_epochs: int = 500, early_stopping_patience: int = None):
        """Class SingleTrainer - a general wrapper for training NNs on
        given datasets, for a given set of hyperparameters.

        :param model: an instance of architecture class inheriting from
        torch.nn.Module, as in amphibian.architectures
        :param batch_size: size of the batch
        :param learning_rate: learning rate for Adam constructor
        :param max_epochs: maximum number of epochs
        :param early_stopping_patience: if not None, a maximum number of epochs
        to wait for validation loss drop before stopping training
        """
        super().__init__()
        # Setting parameters
        self.model = model.to(DEVICE)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience

        # Loss is fixed to nn.CrossEntropyLoss
        self.loss = nn.CrossEntropyLoss()
        # Optimizer is fixed to Adam
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=learning_rate)

    def train(self, train_ds, valid_ds, plot_loss=True, verbose=True,
              save_path=None, need_y: str = 'no'):
        """Method for training, takes train and validation Datasets, as well
        as parameters specifying training monitoring and trains a network for
        a given set of hyperparameters.

        :param train_ds: training Dataset
        :param valid_ds: validation Dataset
        :param plot_loss: whether to plot loss during training
        :param verbose: whether to print loss after each epoch
        :param save_path: if given, serialises the model and saves there
        :param need_y: command to extract y's in order to train Attention based models with
        'state' or 'switch cells' layer
        """
        # Create DataLoaders
        assert need_y in ['no', 'yes'], 'Should be no/yes'
        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              shuffle=True)
        test_dl = DataLoader(valid_ds, batch_size=self.batch_size)

        # Dictionary for losses
        losses = {'train_loss': [], 'valid_loss': []}

        # Plot losses if the user chooses so
        if plot_loss:
            liveloss = PlotLosses()

        # Iterate over epochs
        for epoch in range(self.max_epochs):

            # Switch to training mode
            self.model.train()

            if verbose:
                print('Starting epoch {}'.format(epoch + 1))

            # A list for batch-wise training losses in a given epoch
            epoch_loss = []

            # Iterate over batches
            for idx_batch, batch in enumerate(train_dl):
                self.optimizer.zero_grad()
                if need_y == 'yes':
                    out = self.model(batch[0]['train_obs'].permute(1, 0, 2),
                                     y=batch[1].permute(1, 0))
                    tr_loss = self.loss(out, batch[0]['train_y'].to(DEVICE))
                elif need_y == 'no':
                    out = self.model(batch['train_obs'].permute(1, 0, 2))
                    tr_loss = self.loss(out, batch['train_y'].to(DEVICE))
                epoch_loss.append(tr_loss.item())
                tr_loss.backward()
                self.optimizer.step()

            # Switch to evaluation mode
            self.model.eval()

            # Compute training loss for the epoch
            losses['train_loss'].append(sum(epoch_loss) / len(train_dl))

            # Compute validation loss by iterating through valid dl batches
            with torch.no_grad():

                # A list for batch-wise validation losses
                val_loss = []

                # Iterate over batches in the validation DataLoader
                for idx_v_batch, v_batch in enumerate(test_dl):
                    if need_y == 'yes':
                        val_loss.append(self.loss(
                            self.model(v_batch[0]['test_obs'].permute(1, 0, 2),
                                       y=v_batch[1].permute(1, 0)),
                            v_batch[0]['test_y']).item())
                    elif need_y == 'no':
                        val_loss.append(self.loss(
                            self.model(v_batch['test_obs'].permute(1, 0, 2)),
                            v_batch['test_y']).item())
                losses['valid_loss'].append(sum(val_loss) / len(test_dl))

            # Printing loss for a given epoch
            if verbose:
                print('Loss: {}'.format(losses['valid_loss'][epoch]))
            # Plot loss after each epoch if the user chose to
            if plot_loss:
                logs = {
                    'log_loss': losses['train_loss'][epoch],
                    'val_log_loss': losses['valid_loss'][epoch]
                }

                liveloss.update(logs)
                liveloss.draw()

            # Early stopping
            if self.early_stopping_patience:
                lag_1 = losses['valid_loss'][
                    (epoch - self.early_stopping_patience):epoch
                ]
                lag_2 = losses['valid_loss'][
                    (epoch - self.early_stopping_patience - 1):(epoch - 1)
                ]
                no_drops = sum(True if l1 < l2
                               else False
                               for l1, l2 in zip(lag_1, lag_2))
                if epoch > self.early_stopping_patience and no_drops == 0:
                    break

        # Save last loss
        self.final_loss = np.mean(losses['valid_loss'][-1])
        self.last_epoch = epoch

        # Save model
        if save_path:
            torch.save(self.model.state_dict(), save_path)


class CrossValidation:
    def __init__(self, am_reader, int_start: int, int_end: int, architecture,
                 sampled_param_grid: dict, constant_param_grid: dict,
                 log_path: str, n_iter=100, folds=5, need_y: str = 'no'):
        """Class CrossValidation - hyperparameter optimisation by random search
        and k-fold CV

        :param am_reader: instance of the amphibian.fetch.reader.AmphibianReader
        class
        :param int_start: interval start passed to
        amphibian.preprocess.train_test_split.TrainTestSplit
        :param int_start: interval end passed to
        amphibian.preprocess.train_test_split.TrainTestSplit
        :param architecture: one of the implemented NN architectures.
        :param sampled_param_grid: dictionary of hyperparameters sampled for the
        given CV iteration
        :param constant_param_grid: dictionary of parameters which we want to
        keep fixed for all CV iterations
        :param log_path: where to save a .csv file with CV results
        :param n_iter: number of CV iterations
        :param folds: number of CV folds
        :param need_y: command to extract y's in order to train Attention based models with
        'state' or 'switch cells' layer
        """
        assert architecture in IMPLEMENTED_ARCHITECTURES, \
            'Chosen architecture is not implemented'
        self.am_reader = am_reader
        self.int_start = int_start
        self.int_end = int_end
        self.architecture = architecture
        self.sampled_param_grid = sampled_param_grid
        self.constant_param_grid = constant_param_grid
        self.log_path = log_path \
            + '/cv_log_{:%Y%m%d_%H%M%S}.csv'.format(datetime.now())
        self.n_iter = n_iter
        self.folds = folds
        self.need_y = need_y
        # Dictionary for sampled parameters
        self.sampled_params = {k: [] for k in sampled_param_grid.keys()}
        # Lists for metric statistics and numbers of epochs
        self.results = {'metric_mean': [],
                        'metric_std': [],
                        'metric_min': [],
                        'metric_max': [],
                        'no_epochs': []}

    @staticmethod
    def get_class(cls: str):
        """Method for creating an instance of a class using a string

        :param cls: a string with import path to the class
        :return: an object of the input class
        """
        parts = cls.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m

    @staticmethod
    def create_datasets(self, int_start: int, int_end: int, seq_len: int,
                        need_y: str):
        """Create datasets for all region combinations for given time interval
        beginning and end.

        :param int_start: interval start
        :param int_end: interval start
        :param seq_len: number of days in observation
        :param need_y: extract y's in order to train Attention based models with 'state' or 'switch cells' layer
        :return: train and validation ConcatDatasets
        """
        # Get train test split for selected part of the training set
        input_regs = ['ASIA_PACIFIC', 'ASIA_PACIFIC', 'EMEIA']
        pred_regs = ['EMEIA', 'AMERICA', 'AMERICA']
        train_test_splits = [TrainTestSplit(self.am_reader,
                                            int_start=int_start,
                                            int_end=int_end,
                                            input_reg=ir,
                                            pred_reg=pr)
                             for ir, pr in zip(input_regs, pred_regs)]
        # Prepare dataset
        timeser_datasets = [
            preproc.TimeSeriesDataset(
                tts, int_len=seq_len,
                transform=transforms.Compose([
                    preproc.FillNaN(), preproc.Normalizing(),
                    preproc.DummyFillNaN(), preproc.Formatting(),
                    preproc.FormattingY()
                ]),
                need_y=need_y
            )
            for tts in train_test_splits
        ]
        return torch.utils.data.ConcatDataset(timeser_datasets), \
               torch.utils.data.ConcatDataset(
                   [preproc.ValidDataset(tsds) for tsds in timeser_datasets]
               )

    def run(self):
        """Perform Cross-Validation.

        :return: a pd.DataFrame with CV results
        """
        print('STARTED CROSS-VALIDATION')
        print('Optimizing hyperparameters for {}'.format(self.architecture))

        # CV iterations
        for it in range(self.n_iter):
            print('Beginning CV iteration {:d}'.format(it + 1))

            # Sample parameters
            sampled_params = {}
            for k, v in self.sampled_param_grid.items():
                par = v.rvs(size=1)[0]
                if par.dtype == float:
                    sampled_params[k] = float(par)
                else:
                    sampled_params[k] = int(par)
                self.sampled_params[k].append(par)
            print('Trying for the following parameters: {}'.
                  format(str(sampled_params)))

            # Concatenate sampled and constant parameters
            model_params = {**sampled_params, **self.constant_param_grid}

            # Extract parameters for SingleTrainer
            st_params = {p: model_params.pop(p)
                         for p in NON_MODEL_PARAMETERS}

            # Lists for one-fold losses and epoch numbers before early stopping
            fold_losses, last_epochs = [], []

            # Beginnings and ends for cross-validation intervals
            # One interval is supposed to occupy half of the training set
            # and roll through its entirety
            interval = self.int_end - self.int_start
            delta = np.floor(interval / 2 / (self.folds - 1))
            int_starts = [int(self.int_start + delta * f)
                          for f in range(self.folds)]
            int_ends = [int(self.int_end - delta * (self.folds - f - 1))
                        for f in range(self.folds)]

            # Iterate over folds
            for fold in range(self.folds):
                print('\tFold: {:d}'.format(fold + 1))

                tsds, vds = self.create_datasets(
                    self,
                    int_start=int_starts[fold],
                    int_end=int_ends[fold],
                    seq_len=model_params['seq_len'],
                    need_y=self.need_y
                )

                # Create new instance of model object
                architecture = self.get_class(
                    'amphibian.architectures.' + self.architecture
                )(**model_params)
                # Create new instance of SingleTrainer and begin training
                st = SingleTrainer(model=architecture,
                                   batch_size=model_params['batch_size'],
                                   **st_params)
                st.train(train_ds=tsds, valid_ds=vds, plot_loss=False,
                         need_y=self.need_y)
                last_epochs.append(st.last_epoch)
                print('\tFitting ended after {:d} epochs'.format(
                    st.last_epoch + 1))
                fold_losses.append(st.final_loss)
                print('\tLoss on this fold: {:.5f}'.format(st.final_loss))

            # Summarise computed metrics for a given choice of parameters
            self.results['metric_mean'].append(np.mean(fold_losses))
            self.results['metric_std'].append(np.std(fold_losses))
            self.results['metric_min'].append(min(fold_losses))
            self.results['metric_max'].append(max(fold_losses))
            self.results['no_epochs'].append(last_epochs)
            self.summary_df = pd.concat(
                [pd.DataFrame(self.sampled_params),
                 pd.DataFrame(self.results)],
                axis=1
            )
            self.summary_df.to_csv(self.log_path)
        return self.summary_df


def batch_size_dist(min_num: int, max_num: int):
    """Function for sampling powers of 2.

    :param min_num: minimum number (a power of 2)
    :param max_num: maximum number (a power of 2)
    """
    assert math.log(min_num, 2).is_integer() and math.log(max_num, 2).is_integer(),\
        'Supplied minimum and maximum have to be powers of 2'
    min_pow = int(math.log(min_num, 2))
    max_pow = int(math.log(max_num, 2))
    no = max_pow - min_pow + 1
    return stats.rv_discrete(
        values=([2 ** p for p in np.arange(min_pow, max_pow + 1)],
                [1/no for _ in np.arange(min_pow, max_pow + 1)])
    )
