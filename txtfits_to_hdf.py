import numpy as np
import pandas as pd
from astropy.table import Table
import os
import config
import pyarrow

'''
Visto che gli spettri sono salvati in formati misti (.fits .txt) ho pensato di uniformare il formato. L'output di
questo programma è un file .hdf per ciascuno oggetto.
Oltre alla forma, il programma aggiunge anche tutte le altre info a disposizione, utili per la procedura:
redshift, magnitudine-r, flusso radio a 5ghz, e classificazione CLASS.
Il redshift calcolato in automatico dalla SDSS a volte è impreciso, nel caso in questo renderebbe la procedura di fit
della linea impraticabile, si è sostituito il redshift automatico con quello determinato dal CIV (ispezionando poi lo
spettro per verificarne la correttezza).
I dati dello spettro rest-frame e trimmati, con ivar (riscalato anch'esso per il redshift) sono salvati nel gruppo
"spectrum" del file .hdf.
Le info utili sono salvate nel gruppo "info".
Nei 3 spettri in formato .txt, ivar non è presente. Per consistenza definisco quindi ivar come 1/var in un
intervallo rappresentativo attorno alla linea (1460A-1480A).
'''

class_data = pd.read_csv(config.SDSS_IN_CLASS_FILE, sep='\t', usecols=[0,1,2,3,4])
class_data.columns = ['name', 'radflux', 'magr', 'flag', 'z']

colnames=['flux', 'lambda', 'ivar', 'and_mask', 'or_mask', 'wdisp', 'sky', 'model']

for file in os.listdir(config.FITS_FOLDER):
    obj_name = file.strip('.fits').strip('.txt')
    file_path = os.path.join(config.FITS_FOLDER, file)
    obj_info = class_data[ class_data['name'] == obj_name ]

    print(file)
    z = obj_info['z'].values
    if file.endswith('.fits'):
        tbl = Table.read(file_path, hdu=1)
        obj_spect = tbl.to_pandas()

        # In the DR16 the header is labelled with capital letters
        obj_spect.columns = [c.lower() for c in obj_spect.columns]

        obj_spect.rename(columns = {'loglam':'lambda'}, inplace=True)
        obj_spect.drop(columns=['and_mask', 'or_mask', 'wdisp', 'sky', 'model'], inplace=True)
        obj_spect['lambda'] = (10**obj_spect['lambda'])/(1+z)
        obj_spect['flux'] = obj_spect['flux']*(1+z)

    elif file.endswith('.txt'):
        obj_spect = pd.read_csv(file_path, sep='\t', skiprows=2, header=None)
        obj_spect.columns = ['lambda', 'flux']
        obj_spect['lambda'] = obj_spect['lambda']/(1+z)
        obj_spect['flux'] = obj_spect['flux']*(1+z)
        continuum = obj_spect['flux'][ (obj_spect['lambda']>=1460) & (obj_spect['lambda']<= 1480) ]
        obj_spect['ivar'] = 1/(np.var(continuum.values))

    else: raise

    obj_spect = obj_spect[ (obj_spect['lambda'] >= config.TRIM_INF) & (obj_spect['lambda'] <= config.TRIM_SUP) ]
    obj_spect.reset_index().to_hdf(os.path.join(config.HDF_FOLDER,obj_name), key="spectrum", mode='w')
    obj_info.reset_index().to_hdf(os.path.join(config.HDF_FOLDER,obj_name), key="info", mode='a')
