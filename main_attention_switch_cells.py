import os
import sys

from datetime import datetime
from scipy import stats

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
    'num_layers': 1,
    'max_epochs': 50,
    'early_stopping_patience': 10,
    'recurrent_type': 'lstm',
    'alignment': 'ffnn',
    'additional_y_layer': 'yes',
    'switch_cells': 'yes'
}

if __name__ == '__main__':
    # Declare path to run directory and create it
    run_path = 'data/run_attention_switch_cells{:%Y%m%d_%H%M%S}'.format(datetime.now())
    os.makedirs(run_path)

     # Redirecting outputs
    sys.stdout.flush()
    sys.stderr = open(run_path + '/error_log.txt', 'w')

     # Initialise AmphibianReader
    ar = AmphibianReader('data/all_values/banking',
                         datetime(2010, 7, 16),
                         datetime(2018, 12, 31))

     # Create tensors
    _ = ar.create_torch()

     # Initialise CrossValidation
    cv = CrossValidation(am_reader=ar, int_start=0,
                         int_end=ar.torch['AMERICA'].shape[0],
                         architecture='AttentionModel',
                         sampled_param_grid=SAMPLING_GRID,
                         constant_param_grid=CONSTANT_GRID,
                         log_path=run_path,
                         n_iter=200,
                         need_y='yes',
                         switch_cells='yes')

     # Run CrossValidation
    cv.run()