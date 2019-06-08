import inspect
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from livelossplot import PlotLosses
from scipy import stats
from torch.utils.data import DataLoader

# Set CUDA if available
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


class SingleTrainer:
    def __init__(self, model, batch_size, max_epochs=500,
                 early_stopping_patience=None):
        """
        Class SingleTrainer -

        :param batch_size: size of the batch
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
        self.optimizer = optim.Adam(params=self.model.parameters())

    def train(self, train_ds, valid_ds, plot_loss=True, verbose=True):
        # Define DataLoaders
        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              shuffle=True, drop_last=True)
        test_dl = DataLoader(valid_ds, batch_size=self.batch_size,
                             drop_last=True)

        losses = {'train_loss': [], 'valid_loss': []}

        if plot_loss:
            liveloss = PlotLosses()
        for epoch in range(self.max_epochs):
            if verbose:
                print('Starting epoch {}'.format(epoch))
            epoch_loss = []
            for idx_batch, batch in enumerate(train_dl):
                # Switch to training mode
                self.model.train()
                self.optimizer.zero_grad()
                out = self.model(batch['train_obs'].permute(1, 0 ,2))
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
                val_loss = []
                for idx_v_batch, v_batch in enumerate(test_dl):
                    val_loss.append(self.loss(
                        self.model(v_batch['test_obs'].permute(1, 0, 2)),
                        v_batch['test_y']
                    ).item())
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
        self.final_loss = np.mean(losses[-1]['valid_loss'])
        self.last_epoch = epoch


class CrossValidation:
    def __init__(self, architecture, param_grid: dict, n_iter=100, folds=5):
        """Class hyperparameter optimisation by random search and k-fold CV

        :param architecture: One of the implemented NN architectures.
        :param param_grid: Has to reflect parameters which can be used by the architecture.
        :param n_iter:
        :param folds:
        """
        assert architecture in [name for name, obj in inspect.getmembers(
            'amphibian.architectures'
        )], 'Chosen architecture is not implemented'
        self.architecture = architecture
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.folds = folds
        # Dictionary for sampled parameters
        self.sampled_params = {k: [] for k in param_grid.keys()}
        # Lists for metric statistics and numbers of epochs
        self.results = {'metric_mean': [],
                        'metric_std': [],
                        'metric_min': [],
                        'metric_max': [],
                        'no_epochs': []}

    @staticmethod
    def get_class(cls):
        parts = cls.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m

    def run(self):
        print('STARTED CROSS-VALIDATION')
        print('Optimizing hyperparameters for {}'.format(self.architecture))
        for it in range(self.n_iter):
            print('Beginning CV iteration {:d}'.format(it))
            # Sample a set of hyperparameters
            curr_params = {}
            for k, v in self.param_grid.items():
                par = v.rvs(size=1)[0]
                curr_params[k] = par
                self.sampled_params[k].append(par)
            print('Trying for the following parameters: '.
                  format(str(curr_params)))
            # List for one-fold losses
            fold_losses = []
            # List for last epochs
            last_epochs = []
            for fold in range(self.folds):
                print('\tFold: {:d}'.format(fold))
                architecture = self.get_class(
                    'amphibian.architectures' + self.architecture
                )(**curr_params)
                st = SingleTrainer(model=architecture,
                                   batch_size=curr_params['batch_size'])
                st.train(train_ds=2, valid_ds=2, plot_loss=False)
                last_epochs.append(st.last_epoch)
                print('\tFitting ended after {:d} epochs'.format(st.last_epoch))
                fold_losses.append(st.final_loss)
                print('\tLoss on this fold: {:.5f}'.format(st.final_loss))
            # Summarising computed metrics for a given choice of parameters
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
        return self.summary_df


def batch_size_dist(min, max):
    """Function for sampling powers of 2
    """
    assert math.log(min, 2).is_integer() and math.log(max, 2).is_integer(),\
        'Supplied minimum and maximum have to be powers of 2'
    min_pow = int(math.log(min, 2))
    max_pow = int(math.log(max, 2))
    no = max_pow - min_pow + 1
    return stats.rv_discrete(
        values=([2 ** p for p in np.arange(min_pow, max_pow + 1)],
                [1/no for _ in np.arange(min_pow, max_pow + 1)])
    )
