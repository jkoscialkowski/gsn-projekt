import numpy as np
import os
import pandas as pd
import datetime

from pathlib import Path

DATA_DIR = str(Path(os.path.abspath(__file__)).parents[2]) + '/data/'
ALL_VALUES_DIR = DATA_DIR + 'all_values/'


class AmphibianReader:
    def __init__(self, data_path: str,
                 min_date: datetime.datetime,
                 max_date: datetime.datetime):
        """Class constructor.

        :param data_path:
        :param min_date:
        :param max_date:
        """
        self.data_path = data_path
        self.min_date = min_date
        self.max_date = max_date
        # Read regions data
        regions = pd.read_csv(DATA_DIR + 'regions.csv')
        self.regions = regions.region.unique()

    def read_csvs(self):
        """

        :return:
        """
        self.csvs = {}
        self.quotes = {}
        # Iterate over regions
        for reg in self.regions:
            self.csvs[reg] = []
            self.quotes[reg] = []
            # Iterate over files for this region
            for file in os.listdir(self.data_path + '/' + reg):
                df = pd.read_csv(self.data_path + '/' + reg + '/' + file)
                df = df.assign(
                    Date=pd.to_datetime(df.Date, format='%Y-%m-%d')
                ).drop_duplicates(subset='Date')
                # Gather conditions for min and max dates
                # Initial date for the quote at or before the minimum date?
                min_date_cond = df.Date.iloc[0] <= self.min_date
                # Last date for the quote at or after the maximum date?
                max_date_cond = df.Date.tail(1).iloc[0] >= self.max_date
                if min_date_cond and max_date_cond:
                    self.csvs[reg].append(df)
                    self.quotes[reg].append(file)
        return self.csvs

    def get_unique_dates(self):
        """Extract unique dates from all csvs.

        These are necessary, because trade does not occur on all days in all
        countries and the data size has to be unified somehow.

        :return:
        """
        # Initialize an empty set
        unique_dates = set()
        # Iterate over regions
        for reg in self.regions:
            # Iterate over files for this region
            for file in self.csvs[reg]:
                unique_dates = unique_dates.union(set(file.Date))
            self.unique_dates = pd.DataFrame(
                {'Date':list(unique_dates)}
            ).sort_values('Date')
        return self.unique_dates

    def create_numpy(self):
        """Create 3-dim (date x prices x company) numpy arrays for each region.

        Data is left joined to unique dates so that the date range is the same
        for all companies, possibly filled with NaN's.

        :return:
        """
        self.numpy = {}
        # Iterate over regions
        for reg in self.regions:
            array_list = []
            # Iterate over files for this region
            for df in self.csvs[reg]:
                # Left join the current dataframe to the unique date dataframe
                joined_df = self.unique_dates.set_index('Date').join(
                    df.set_index('Date'),
                    how='left'
                ).reset_index().drop('Date', axis=1)
                array = np.expand_dims(joined_df.values, 2)
                array_list.append(array)
            self.numpy[reg] = np.concatenate(array_list, axis=2)
        return self.numpy

    #def create_torch(self):

