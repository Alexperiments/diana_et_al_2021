import config
import utils
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json


def gaussian1(x, A, mean1, sigma1):
    return A*np.exp(-np.power((x-mean1)/sigma1,2.) / 2.)

def gaussian2(x, A, mean1, sigma1, B, mean2, sigma2):
    return (A*np.exp(-np.power((x-mean1)/sigma1,2.) / 2.) +
            B*np.exp(-np.power((x-mean2)/sigma2,2.) / 2.))

def gaussian3(x, A, mean1, sigma1, B, mean2, sigma2, C, mean3, sigma3):
    return (A*np.exp(-np.power((x-mean1)/sigma1,2.) / 2.) +
            B*np.exp(-np.power((x-mean2)/sigma2,2.) / 2.) +
            C*np.exp(-np.power((x-mean3)/sigma3,2.) / 2.))

init_guess1 = [ 1, 1549, 30 ]
init_guess2 = [ 1, 1549, 30, 1, 1549, 30 ]
init_guess3 = [ 1, 1549, 30, 1, 1549, 30, 1, 1549, 30 ]


class Choose_a_fit:
    def __init__(self, fig):
        self.continuum = int_dict['continuum']
        self.masks = []
        self.cid2 = fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid2 = fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig = fig
        self.ax = fig.gca()
        self.fit_lines = self.ax.lines[5:]

    def on_key(self, event):
        if event.key == '1':
            pass
        elif event.key == '2':
            pass
        elif event.key == '3':
            pass
        elif event.key == 'f4':
            if (len(self.continuum) == 4) and (len(self.masks)%2 == 0):
                int_dict['continuum'] = str(self.continuum)
                int_dict['masks'] = str(self.masks)
                name = obj_name + '.png'
                plt.savefig(config.FITTED_PLOTS+name, dpi=300)
                plt.close(self.fig)
            else:
                print(
                f"Punti continuo: {len(self.continuum)}/4\tPunti maschera: {len(self.masks)}"
                )

    def on_click(self, event):
        if self.continuum_mode:
            if len(self.continuum) < 4:
                self.continuum.append(event.xdata)
            else: print("Ho già 4 punti per il continuo, rimuoverne qualcuno!")
        elif self.masks_mode: self.masks.append(event.xdata)
        self.update()

    def on_close(self, event):
        del self.continuum
        del self.masks

    def update(self):
        for line in self.continuum:
            self.ax.axvline(line, color='green', ls='--')
        if (len(self.masks)%2 == 0) & (len(self.masks) != 0):
            self.spans.append(self.ax.axvspan(self.masks[-2], self.masks[-1], alpha=0.5, color='gray'))
        if len(self.continuum) == 4:
            m, q = continuum_fit(obj_spectra, self.continuum)
            int_dict['m'] = m
            int_dict['q'] = q
            x_bin = np.arange(self.continuum[0], self.continuum[3], 1)
            self.ax.plot(x_bin, q + m*x_bin, color='red')
        self.fig.canvas.draw()


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

    par1, pcov1 = curve_fit(gaussian1, lam, flux, p0=init_guess1, sigma=np.sqrt(1/ivar))
    par2, pcov2 = curve_fit(gaussian2, lam, flux, p0=init_guess2, sigma=np.sqrt(1/ivar))
    par3, pcov3 = curve_fit(gaussian3, lam, flux, p0=init_guess3, sigma=np.sqrt(1/ivar))

    for i in range(len(masks)//2):
        plt.axvspan(masks[i*2], masks[i*2+1], alpha=0.5, color='gray')
    for line in cont_lines:
        plt.axvline(line, ls='--', color='green')

    x_fit = np.arange(min(lam), max(lam), 1)
    plt.plot(x_fit, gaussian(x_fit, *par), color='red', zorder=10)
    plt.plot(lam, flux, color='black', lw=0.5)
    plt.axhline(0)
    plt.show()
