from amphibian.fetch.reader import AmphibianReader
import torch
import datetime

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
                                if torch.isnan(self.torch[reg][i - 1, j, k]) != 1:
                                    self.torch[reg][i, j, k] = self.torch[reg][i - 1, j, k]
        return self.torch

    # function, which computes returns
    def compute_returns(self, method = 'day_before'):
        """Creating tensor with return

        :return: self.torch_returns - tensor with returns"""

        self.torch_returns = self.create_torch()
        self.fill_nan()

'''TO DO:
1. compute_returns
2. Dataloader
3. Dataset
4. Multiprocessing
'''