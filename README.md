# Amphibian
Cross-timezone, business-wide share price prediction using various recurrent architectures. 
Can Audi share prices be predicted with Asian motor company stock quotes? Price prediction using Attention neural network.

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