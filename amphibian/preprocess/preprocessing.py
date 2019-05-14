from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class TimeSeriesDataset(Dataset):
    def __init__(self, amReader, int_len = 30, input_reg = 'ASIA_PACIFIC', pred_reg = 'EMEIA'):
        self.amReader = amReader
        self.len = self.amReader.torch[input_reg].size()[0] - int_len
        self.observation = {}
        self.y = {}
        sample_no = 0
        for i in range(self.len):
            self.observation[i] = self.amReader.torch[input_reg][i:i + int_len, :, :]
            self.y[i] = self.amReader.torch[pred_reg][i + int_len - 1, 5, 0] # we want to predict Adj Close price

    def __len__(self): # return the length of the data
        return self.len

    def __getitem__(self, item): # return one item on the index
        return self.observation[item], self.y[item]



'''TO DO:
1. Compose transforms - filling nan -> scalling -> train_test split
2. compute_returns
2. fill_nan
3. scalling
4. train_test spli

Zalety podklasy:
1. daje dostep do wszystkich metod i atrybutow
2. wtedy processing mialby brac obiekt z klasy AmphibianReader i na tym obiekcie 

Pytania:
1. Co w naszym przypadku będzie batchem
'''

'''
- patrzymy po branzy
- liczenie srednich i odchylen na training set i pozniej aplikacja do test set
- chce robic prognoze jednej spolki
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

