import datetime
import numpy as np
import os
import pandas as pd
import torch

from pathlib import Path

DATA_DIR = str(Path(os.path.abspath(__file__)).parents[2]) + '/data/'
ALL_VALUES_DIR = DATA_DIR + 'all_values/'


class AmphibianReader:
    def __init__(self, data_path: str,
                 min_date: datetime.datetime,
                 max_date: datetime.datetime):
        """Class for reading quotes and transforming them to numpy arrays
        and torch tensors.

        :param data_path: Path to folder with region folders containing quotes
        :param min_date: Minimum date to be read
        :param max_date: Maximum date to be read
        """
        self.data_path = data_path
        self.min_date = min_date
        self.max_date = max_date
        # Read regions data
        regions = pd.read_csv(DATA_DIR + 'regions.csv')
        self.regions = regions.region.unique()

    def read_csvs(self):
        """Read files as .csv's from the specified `data_path` and apply
        date filters. Date column is cast to datetime.datetime and duplicates
        are dropped.

        :return: Dictionary with region:quote:pd.DataFrame of quotes
        """
        self.csvs = {}
        # Iterate over regions
        for reg in self.regions:
            self.csvs[reg] = {}
            # Iterate over files for this region
            for file in os.listdir(self.data_path + '/' + reg):
                df = pd.read_csv(self.data_path + '/' + reg + '/' + file)
                # Cast Date to datetime.datetime and drop duplicates
                df = df.assign(
                    Date=pd.to_datetime(df.Date, format='%Y-%m-%d')
                ).drop_duplicates(subset='Date')
                # Gather conditions for min and max dates
                # Initial date for the quote at or before the minimum date?
                min_date_cond = df.Date.iloc[0] <= self.min_date
                # Last date for the quote at or after the maximum date?
                max_date_cond = df.Date.tail(1).iloc[0] >= self.max_date
                if min_date_cond and max_date_cond:
                    self.csvs[reg][file] = df
        return self.csvs

    def get_unique_dates(self):
        """Extract unique dates from all csvs.

        These are necessary, because trade does not occur on all days in all
        countries and the data size has to be unified somehow.

        :return: One-column pd.DataFrame with unique dates
        """
        # Initialize an empty set
        unique_dates = set()
        # Iterate over regions
        for reg in self.regions:
            # Iterate over files for this region
            for file in self.csvs[reg].values():
                # Add unseen dates to the set
                unique_dates = unique_dates.union(set(file.Date))
            # Convert to dataframe to facilitate further joining
            self.unique_dates = pd.DataFrame(
                {'Date':list(unique_dates)}
            ).sort_values('Date')
        return self.unique_dates

    def create_numpy(self):
        """Create 3-dim (date x prices x company) numpy arrays for each region.

        Data is left joined to unique dates so that the date range is the same
        for all companies, possibly filled with NaN's.

        :return: Dictionary region:np.array with quotes
        """
        # Checking if the necessary objects exist
        if not hasattr(self, 'csvs'):
            self.read_csvs()
        if not hasattr(self, 'unique_dates'):
            self.get_unique_dates()
        self.numpy = {}
        # Iterate over regions
        for reg in self.regions:
            array_list = []
            # Iterate over files for this region
            for df in self.csvs[reg].values():
                # Left join the current dataframe to the unique date dataframe
                joined_df = self.unique_dates.set_index('Date').join(
                    df.set_index('Date'),
                    how='left'
                ).reset_index().drop('Date', axis=1)
                # Convert to a 3-dim array and append to list
                array = np.expand_dims(joined_df.values, 2)
                array_list.append(array)
            self.numpy[reg] = np.concatenate(array_list, axis=2)
        return self.numpy

    def create_torch(self):
        """Create 3-dim (date x prices x company) torch tensors for each region.
        The tensors are created from the numpy arrays. Type is cast to
        torch.float32 and device is set according to CUDA availability.

        :return: Dictionary region:torch.Tensor with quotes
        """
        # Checking if the numpy arrays have been created
        if not hasattr(self, 'numpy'):
            self.create_numpy()
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.torch = {}
        for reg, nparr in self.numpy.items():
            self.torch[reg] = torch.from_numpy(nparr).to(dtype=torch.float32,
                                                         device=device)
        return self.torch

    def __getitem__(self, what):
        """Method for extracting read data using square brackets.

        :param what: Either a string or a tuple with selected keys
        :return: Selected dictionary/pd.DataFrame/np.array/torch.Tensor
        """
        # Selecting using the first element of `what`
        if len(what) == 1:
            return getattr(self, what)
        else:
            result = getattr(self, what[0])

        # Further selection
        for arg in what[1:]:
            result = result[arg]
        return result


