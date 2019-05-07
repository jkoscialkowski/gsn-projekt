import pandas as pd
import pandas_datareader as pd_datareader
import os
import argparse

DATA_DIR = '../../data/'
ALL_VALUES_DIR = '../../data/all_values/'


def download(quote_type: str, exchange: str, business: str):
    if not os.path.exists(ALL_VALUES_DIR):
        os.mkdir(ALL_VALUES_DIR)
    if not os.path.exists(ALL_VALUES_DIR + quote_type):
        os.mkdir(ALL_VALUES_DIR + quote_type)
    if not os.path.exists(ALL_VALUES_DIR + quote_type + '/' + exchange):
        os.mkdir(ALL_VALUES_DIR + quote_type + '/' + exchange)
    tickers = pd.read_csv(DATA_DIR + 'tickers_' + quote_type + '.csv')
    # Creating logical vector depending on selected exchange & business
    tickers_logical = tickers['Exchange'] == exchange
    if quote_type == 'stocks' and business:
        tickers_logical = (
            tickers_logical & (tickers['Category Name'] == business)
        )
    for ticker in tickers.loc[tickers_logical, 'Ticker']:
        print(ticker)
        try:
            pd_datareader.DataReader(ticker, 'yahoo').to_csv(ALL_VALUES_DIR + quote_type + '/' + exchange + '/' + ticker)
        except:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--Type',
                        help='Export values of selected type')
    parser.add_argument('-E', '--Exchange',
                        help='Export values from selected exchange')
    parser.add_argument('-B', '--Business',
                        help='Export values for selected business')
    args = parser.parse_args()
    download(args.Type, args.Exchange, args.Business)
