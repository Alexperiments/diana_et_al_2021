import config
import utils
import intervals
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def continuum_estimate(x, y, inf1, inf2, sup1, sup2):
    mask = [((x >= sup1) & (x <= sup2)) | ((x >= inf1) & (x <= inf2))]
    yt = y[mask]
    xy = x[mask]
    return np.polyfit(xt, yt, deg=1)

def mask(spectrum, masks):
    if not masks: return spectrum
    for j in range(int(len(masks)/2)):
        low = masks[2*j]
        high = masks[2*j+1]
        spectrum = spectrum[(spectrum["lambda"] <= low) | (spectrum["lambda"] >= high)]
    return spectrum[spectrum["ivar"] != 0]

def init_dictionary(name):
    bg_window = intervals.bg_dict.get(name)
    if not bg_window: bg_window = [1440,1460,1690,1710]
    mask_window = intervals.mask_dict.get(name)
    #bounds = intervals.bound_dict_lowz.get(name)
    #if not bounds:
    bounds = [[0,1510,0,0,1510,0],[2000,1570,50,2000,1570,50]]
    return bg_window, mask_window, bounds

def gaus_func(x, a, x1, sigma1, b=0, x2=0, sigma2=1):
    return  (abs(a)*np.exp(-(x-x1)**2/(2*sigma1**2))+
            abs(b)*np.exp(-(x-x2)**2/(2*sigma2**2)))

for obj_name in os.listdir(config.HDF_FOLDER):
    #obj_name = 'GB6J130603+552947'
    file_path = os.path.join(config.HDF_FOLDER, obj_name)
    print(obj_name)
    obj_spectra = pd.read_hdf(file_path, key='spectrum')
    obj_info = pd.read_hdf(file_path, key='info')

    bg_window, mask_window, bounds = init_dictionary(obj_name)

    p_cont,  = continuum_estimate(obj_spectra["lambda"], obj_spectra["flux"], *bg_window)

    plt.plot(obj_spectra["lambda"], obj_spectra["flux"], color='black', lw=0.5)
    plt.plot(obj_spectra["lambda"], m*obj_spectra["lambda"] + q, color='red', lw=0.5)
    plt.show()

    obj_spectra["flux"] = obj_spectra["flux"] - (m*obj_spectra["lambda"] + q)
    obj_spectra = mask(obj_spectra, mask_window)
    obj_spectra["error"] = np.sqrt(1/obj_spectra["ivar"])*(1+obj_info["z_temp"].values)

    init_guess = [10, 1549, 10, 10, 1549, 15]
    par, cov = curve_fit(
        f=gaus_func,
        xdata=obj_spectra["lambda"],
        ydata=obj_spectra["flux"],
        sigma=obj_spectra["error"],
        p0=init_guess,
        bounds=bounds
    )

    plt.matshow(cov)
    plt.show()
    break

    plt.plot(obj_spectra["lambda"], obj_spectra["flux"], color='black', lw=0.5)
    plt.plot(obj_spectra["lambda"], obj_spectra["error"])
    plt.plot(obj_spectra["lambda"], gaus_func(obj_spectra["lambda"], *par), color='red')
    plt.xticks(np.arange(config.TRIM_INF, config.TRIM_SUP+10, 10))
    plt.grid(color='gray', linestyle='--')
    plt.savefig(f"check_plot/{obj_name}.png", dpi=300)
    plt.cla()
