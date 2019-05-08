import argparse
import os
import pandas as pd

from pathlib import Path

from amphibian.fetch.downloader import download

DATA_DIR = str(Path(os.path.abspath(__file__)).parents[2]) + '/data/'
ALL_VALUES_DIR = DATA_DIR + 'all_values/'

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--Type',
                        help='Export values of selected type')
    parser.add_argument('-B', '--Business',
                        help='Export values for selected business')
    args = parser.parse_args()

    # Reading region data
    regions = pd.read_csv(DATA_DIR + 'regions.csv')

    # Downloading data
    for r in regions.region.unique():
        download(args.Type, r, args.Business)


