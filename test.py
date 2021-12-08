import numpy as np
from extinction import fitzpatrick99 as fp99
from extinction import remove as ext_remove
import pandas as pd
import config
import os
from astropy.table import Table
import matplotlib.pyplot as plt

class_data = pd.read_csv('data/class_in_sdss_new.txt', sep='\s+')

colnames=['flux','lambda','ivar','and_mask','or_mask','wdisp','sky','model']

with_spect = os.listdir(config.SOURCE_FOLDER)
with_spect = [a.strip('.fits').strip('.txt') for a in with_spect]

class_data = class_data.query('classname in @with_spect')
sorted_data = class_data.sort_values(['a_v', 'classname'])

mass_diff = []
zs = []

for file in os.listdir(config.SOURCE_FOLDER):
    obj_name = file.strip('.fits').strip('.txt')
    if obj_name not in class_data['classname'].values: continue

    print(obj_name)
    file_path = os.path.join(config.SOURCE_FOLDER, file)
    target_path = os.path.join(config.OUT_FOLDER + obj_name)

    obj_info = class_data[ class_data['classname'] == obj_name]

    z = obj_info['z'].values
    a_v = obj_info['a_v'].values

    if file.endswith('.fits'):
        tbl = Table.read(file_path, hdu=1)
        obj_spect = tbl.to_pandas()

        # In the DR16 the header is labelled with capital letters
        obj_spect.columns = [c.lower() for c in obj_spect.columns]

        obj_spect.rename(columns={'loglam': 'lambda'}, inplace=True)
        obj_spect['lambda'] = 10 ** obj_spect['lambda']

    elif file.endswith('.txt'):
        obj_spect = pd.read_csv(file_path, sep='\t', skiprows=2, header=None)
        obj_spect.columns = ['lambda', 'flux']

    low_trim = 1350*(1+z[0])
    obj_spect = obj_spect[obj_spect['lambda'] >= low_trim]
    extinction_curve = fp99(obj_spect['lambda'].to_numpy(dtype=np.double), a_v, 3.1)
    dered_spectra = ext_remove(extinction_curve, obj_spect['flux'])

    diff = dered_spectra.to_numpy()[0]/obj_spect['flux'].to_numpy()[0]
    mass_diff.append(.53*np.log10(diff))
    zs.append(z[0])

plt.scatter(zs, mass_diff, s=1, color='black')
plt.xlabel('z')
plt.ylabel(r'$\Delta$M (log)')
plt.show()