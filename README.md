# Amphibian
Can Audi share prices be predicted with Asian motor company stock quotes? Price predition using Attention neural network.

## Run on your own
This guide assumes you have Python 3.x installed on your computer. Clone the repository to a desired local directory

```
git clone https://github.com/jkoscialkowski/gsn-projekt.git
```

Then `cd` to this folder, create and activate a virtualenv and run 

```
pip install -r requirements.txt
```

Afterwards simply run

```
python -m amphibian.app [fetch_new_data] [train network]
```