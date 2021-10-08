import config
import pandas as pd
import numpy as np
import os
from astropy.table import Table

class_data = pd.read_csv(config.SDSS_IN_CLASS_FILE, sep='\t', usecols=[0,1,2,3,4])
class_data.columns = ['name', 'radflux', 'magr', 'flag', 'z']

for file in os.listdir(config.FITS_FOLDER):
    if file.endswith('.fits'):
        obj_name = file.strip('.fits').strip('.txt')
        file_path = os.path.join(config.FITS_FOLDER, file)

        tbl = Table.read(file_path, hdu=3)
        info = tbl.to_pandas()
        z_CIV = info.loc[2, ["LINEZ"]].values

        z_best = class_data['z'][ class_data['name'] == obj_name ].values

        delta = z_CIV - z_best
        if delta>=0.01: print(f"{obj_name} {z_best} -> {z_CIV}")
