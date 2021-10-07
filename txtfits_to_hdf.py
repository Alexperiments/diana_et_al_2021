import numpy as np
import pandas as pd
from astropy.table import Table
import os
import config
import pyarrow

class_data = pd.read_csv(config.SDSS_IN_CLASS_FILE, sep='\t', usecols=[0,1,2,3,4])
class_data.columns = ['name', 'radflux', 'magr', 'flag', 'z']

colnames=['flux', 'lambda', 'ivar', 'and_mask', 'or_mask', 'wdisp', 'sky', 'model']

for file in os.listdir(config.FITS_FOLDER):
    obj_name = file.strip('.fits').strip('.txt')
    file_path = os.path.join(config.FITS_FOLDER, file)
    obj_info = class_data[ class_data['name'] == obj_name ]
    z = obj_info['z_temp'].values
    print(file)
    if file.endswith('.fits'):
        tbl = Table.read(file_path, hdu=1)
        obj_spect = tbl.to_pandas()
        obj_spect.rename(columns = {'loglam':'lambda'}, inplace=True)
        obj_spect.drop(columns=['and_mask', 'or_mask', 'wdisp', 'sky', 'model'], inplace=True)
        obj_spect['lambda'] = (10**obj_spect['lambda'])/(1+z)

    '''elif file.endswith('.txt'):
        print(file)
        obj_spect = pd.read_csv(file_path, sep='\t', skiprows=2, header=None)
        print(obj_spect)
        obj_spect.columns = ['lambda', 'flux']
        obj_spect['lambda'] = obj_spect['lambda']/(1+z)'''

    obj_spect['flux'] = obj_spect['flux']*(1+z)
    obj_spect = obj_spect[ (obj_spect['lambda'] >= config.TRIM_INF) & (obj_spect['lambda'] <= config.TRIM_SUP) ]
    obj_spect.reset_index().to_hdf(os.path.join(config.HDF_FOLDER,obj_name), key="spectrum", mode='w')
    obj_info.reset_index().to_hdf(os.path.join(config.HDF_FOLDER,obj_name), key="info", mode='a')
