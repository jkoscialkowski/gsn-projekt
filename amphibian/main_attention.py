import argparse
import os
import sys

from datetime import datetime
from scipy import stats

from amphibian.fetch.downloader import download
from amphibian.fetch.reader import AmphibianReader
from amphibian.train import CrossValidation, batch_size_dist

# Define parameter grids
SAMPLING_GRID = {
    'learning_rate': stats.uniform(1e-4, 1e-2),
    'batch_size': batch_size_dist(32, 256),
    'seq_len': stats.randint(5, 30),
    'hidden_size': stats.randint(5, 20),
    'dropout': stats.uniform(0, 0.5)
}

CONSTANT_GRID = {
    'input_size': 60,
    'n_outputs': 3,
    'num_layers': 2,
    'max_epochs': 5,
    'early_stopping_patience': 10,
    'recurrent_type': 'lstm',
    'alignment': 'ffnn'
}

if __name__ == '__main__':
    # Parse script input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--Arch',
                        help='Use chosen architecture')
    args = parser.parse_args()

    # Declare path to run directory and create it
    run_path = 'data/run_attention_{%Y%m%d_%H%M%S}'.format(datetime.now())
    os.makedirs(run_path)

    # Redirecting outputs
    sys.stdout.flush()
    sys.stderr = open(run_path + '/error_log.txt')

    # Initialise AmphibianReader
    ar = AmphibianReader(DATA_PATH,
                         datetime.datetime(2011, 1, 1),
                         datetime.datetime(2018, 1, 1))

    # Initialise CrossValidation
    cv = CrossValidation(ar, 0, 1000, 'AttentionModel',
                         SAMPLING_GRID, CONSTANT_GRID, run_path)

    # Run CrossValidation
    cv.run()