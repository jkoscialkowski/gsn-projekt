# Amphibian
Cross-timezone, business-wide share price prediction using various recurrent 
architectures. 

The goal of the project was to investigate whether stock quotes for companies 
from the same area, but earlier timezone carry _decent_ predictive power if
compared to the company itself or all companies from its region. Code in this 
repository allows the user to download data for a chosen business and then
experiment with several RNN-based architectures.


## Run on your own
This guide assumes you have Python 3.x installed on your computer. Clone the repository to a desired local directory

```bash
git clone https://github.com/jkoscialkowski/gsn-projekt.git
```

Then `cd` to this folder, create and activate a virtualenv and run 

```bash
pip install -r requirements.txt
```

Afterwards run  
```bash
python3 amphibian/fetch/downloader.py -T banking -R REGION -B banking
```
to download the data. You need to substitute REGION with all of the
following, one-by-one: ASIA_PACIFIC, EMEIA, AMERICA.

Finally, you can run one of the `main*.py` files to start cross-validation.
For example to train the Attention network run
```
python -m main_attention.py 
```

This will yield .csv files with Cross-Validation results. The best set of
hyperparameters can then be used to train a final model which will be used for
generating insights.

## Visualise
Amphibian comes with the `amphibian.visual` module which implements:
* a nice class for confusion matrix display (with precisions, recalls and 
accuracy presented),
* Model-Agnostic Variable Importance which computes permutational importance for
a given company and shows the results on a barplot. 

We advise playing around with a trained model and the visualisation tools 
implemented in the said module. 