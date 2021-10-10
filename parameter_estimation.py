import config
import utils
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json


def gaussian(x, A, mean1, sigma1, B, mean2, sigma2):
    return (A*np.exp(-np.power((x-mean1)/sigma1,2.) / 2.) +
            B*np.exp(-np.power((x-mean2)/sigma2,2.) / 2.))
            #C*np.exp(-np.power((x-mean3)/sigma3,2.) / 2.)



for i, obj_name in enumerate(sorted(os.listdir(config.HDF_FOLDER))):
    file_path = os.path.join(config.HDF_FOLDER, obj_name)
    print(i, obj_name)
    obj_spectra = pd.read_hdf(file_path, key='spectrum')
    obj_info = pd.read_hdf(file_path, key='info')
    obj_pre_fit = pd.read_hdf(file_path, key='pre_fit')

    flux = obj_spectra['flux'].values
    lam = obj_spectra['lambda'].values
    ivar = obj_spectra['ivar'].values
    m = obj_pre_fit['m'].values
    q = obj_pre_fit['q'].values
    masks = json.loads(obj_pre_fit['masks'].values[0])
    cont_lines = json.loads(obj_pre_fit['continuum'].values[0])

    trim = (lam >= 1450) & (lam <= 1650)
    lam = lam[trim]
    flux = flux[trim]
    ivar = ivar[trim]

    masks.append(1600)
    masks.append(1680)
    mask_condition = [True for _ in range(len(lam))]
    for i in range(len(masks)//2):
        mask_condition = ((lam < masks[i*2]) | (lam > masks[i*2+1])) & mask_condition

    flux = flux[mask_condition]
    lam = lam[mask_condition]
    ivar = ivar[mask_condition]

    flux = flux - (m*lam + q)
    init_guess = [
        1, 1549, 30,
        1, 1549, 30,
        #0, 1549, 1
    ]
    bounds = [
        [0, 1510, 0, 0, 1510, 0],
        [np.inf, 1580, np.inf, np.inf, 1580, np.inf]
    ]
    par, pcov = curve_fit(gaussian, lam, flux, p0=init_guess, sigma=np.sqrt(1/ivar), bounds=bounds)

    for i in range(len(masks)//2):
        plt.axvspan(masks[i*2], masks[i*2+1], alpha=0.5, color='gray')
    for line in cont_lines:
        plt.axvline(line, ls='--', color='green')

    x_fit = np.arange(min(lam), max(lam), 1)
    plt.plot(x_fit, gaussian(x_fit, *par), color='red', zorder=10)
    plt.plot(lam, flux, color='black', lw=0.5)
    plt.axhline(0)
    plt.show()
