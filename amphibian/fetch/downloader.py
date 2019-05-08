import pandas as pd
import pandas_datareader as pd_datareader
import os
import argparse

from pathlib import Path

DATA_DIR = str(Path(os.path.abspath(__file__)).parents[2]) + '/data/'
ALL_VALUES_DIR = DATA_DIR + 'all_values/'


def download(quote_type: str, region: str, business: str = None):
    """Function for quote downloading.

    :param quote_type:
    :param region:
    :param business:
    :return:
    """
    # Create necessary high-level paths
    if not os.path.exists(ALL_VALUES_DIR):
        os.mkdir(ALL_VALUES_DIR)
    if not os.path.exists(ALL_VALUES_DIR + quote_type):
        os.mkdir(ALL_VALUES_DIR + quote_type)

    # Read selected tickers and regions data
    tickers = pd.read_csv(DATA_DIR + 'tickers_' + quote_type + '.csv')
    regions = pd.read_csv(DATA_DIR + 'regions.csv')

    # Consider business for stocks or not
    tickers_logical = tickers['Exchange'].isin(
        regions.loc[regions.region == region, 'Exchange'].values
    )
    if quote_type == 'stocks' and business:
        quotes_path = ALL_VALUES_DIR + quote_type + '/' + business + '/' + region
        tickers_logical = (
            tickers_logical & (tickers['Category Name'] == business)
        )
    else:
        quotes_path = ALL_VALUES_DIR + quote_type + '/' + region

    # Create directory for storing selected quotes
    if not os.path.exists(quotes_path):
        os.makedirs(quotes_path)

    # Attempt to download data
    for ticker in tickers.loc[tickers_logical, 'Ticker']:
        print(ticker)
        try:
            pd_datareader.DataReader(ticker, 'yahoo').to_csv(
                quotes_path + '/' + ticker
            )
        except:
            pass


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--Type',
                        help='Export values of selected type')
    parser.add_argument('-R', '--Region',
                        help='Export values from selected region')
    parser.add_argument('-B', '--Business',
                        help='Export values for selected business')
    args = parser.parse_args()
    download(args.Type, args.Region, args.Business)
