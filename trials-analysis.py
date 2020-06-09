import pickle
import hyperopt
import pandas as pd
from pprint import pprint

BASE_DIR = 'data'
FILE = f'{BASE_DIR}/trials.pkl'

with open(FILE, 'rb') as f:
    trials = pickle.load(f)

pprint(dir(trials))
pprint(trials.best_trial)
df = pd.DataFrame(trials)
pprint(df.columns)
