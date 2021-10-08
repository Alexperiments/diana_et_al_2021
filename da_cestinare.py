import pandas as pd
import matplotlib.pyplot as plt
import os

obj_spect = pd.read_csv("spectra_fits/GB6J205332+010307.txt", sep='\t', skiprows=2, header=None)
obj_spect[1] = obj_spect[1]*1e17
obj_spect.to_csv("spectra_fits/GB6J205332+0103071.txt", sep='\t', index=False)

plt.plot(obj_spect[0], obj_spect[1], lw=0.5, color='black')
plt.show()

'''files = os.listdir("spectra_fits/")
for file in files:
    if file.endswith(".txt"):
        print(file)
        obj_spect = pd.read_csv("spectra_fits/"+file, sep='\t', skiprows=2, header=None)
        obj_spect.columns = ['lambda', 'flux']

        plt.plot(obj_spect['lambda'], obj_spect['flux'], lw=0.5, color='black')
        plt.show()'''
