import config
import pandas as pd
import numpy as np
import os
from astropy.table import Table

for file in os.listdir(config.FITS_FOLDER):
    if file.endswith('.fits'):
        obj_name = file.strip('.fits').strip('.txt')
        file_path = os.path.join(config.FITS_FOLDER, file)

        tbl = Table.read(file_path, hdu=3)
        info = tbl.to_pandas()
        z_CIV = info.loc[2, ["LINEZ"]].values

        tbl = Table.read(file_path, hdu=2)
        names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
        info = tbl[names].to_pandas()
        z_best = info["Z"].values

        delta = z_CIV - z_best
        if delta>=0.002: print(obj_name, z_best, z_CIV)
