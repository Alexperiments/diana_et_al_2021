import numpy as np
from extinction import fitzpatrick99 as fp99
from extinction import remove as ext_remove
import pandas as pd
import config
import pickle
import os
from astropy.table import Table
import matplotlib.pyplot as plt

with open(config.PARAMETERS_FILE, 'rb') as f:
    param = pickle.load(f)

with open('data/parameters_old.pkl', 'rb') as f:
    param_old = pickle.load(f)

old_select = pd.read_csv('data/selection_old.txt', sep='\t', index_col='classname')
select = pd.read_csv('data/selection.txt', sep='\t', index_col='classname')

for row in old_select.iterrows():
    select.drop(index=row[0], errors='ignore', inplace=True)

print(select)
