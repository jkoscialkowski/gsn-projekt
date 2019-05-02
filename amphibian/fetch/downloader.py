import pandas as pd
import pandas_datareader as pd_datareader
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--Type', help='Export values of selected type')
    parser.add_argument('-E', '--Exchange', help='Export values from selected exchange')
    args = parser.parse_args()
    data_dir = './data/'
    all_values_dir = './data/all_values/'
    if not os.path.exists(all_values_dir):
        os.mkdir(all_values_dir)
    if not os.path.exists(all_values_dir + args.Type):
        os.mkdir(all_values_dir + args.Type)
    if not os.path.exists(all_values_dir + args.Type + '/' + args.Exchange):
        os.mkdir(all_values_dir + args.Type + '/' + args.Exchange)
    tickers = pd.read_csv(data_dir + 'tickers_' + args.Type + '.csv')
    for ticker in tickers[tickers['Exchange'] == args.Exchange]['Ticker']:
        try:
            pd_datareader.DataReader(args.Exchange, 'yahoo').to_csv(all_values_dir + args.Type + '/' + args.Exchange + '/' + ticker)
        except:
            pass