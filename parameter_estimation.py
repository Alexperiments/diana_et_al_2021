import config
import utils
from utils import gaussian3
import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle
import math


def calc_line_dispersion(A, mean1, sigma1, B=0, mean2=0, sigma2=1, C=0, mean3=0, sigma3=1):
    area = (A*sigma1 + B*sigma2 + C*sigma3)
    integral = math.sqrt(2*math.pi)*area
    centroid = (A*mean1*sigma1 + B*mean2*sigma2 + C*mean3*sigma3)/area
    variance = ((mean1**2+sigma1**2)*A*sigma1 + (mean2**2+sigma2**2)*B*sigma2 + (mean3**2+sigma3**2)*C*sigma3)/area - centroid*centroid
    return integral, np.sqrt(variance)

def calc_fwhm(x,ym):
    max_y = max(ym)  # Find the maximum y value
    xs = x[ ym >= max_y/2.]
    return (max(xs) - min(xs))

def flux_to_lum(flux, dl):
    return np.log10(flux) -17 + np.log10(4*math.pi) + 2*np.log10(dl*3.086) + 48

def calc_mass_VP06(velocity, lamL1350, flag):
    if flag=='sigma': return 6.73 + 2*np.log10(velocity/1000) + 0.53*(lamL1350-44)
    if flag=='fwhm': return 6.66 + 2*np.log10(velocity/1000) + 0.53*(lamL1350-44)

def calc_edd_ratio(lamL1350, mass):
    Lbol = lamL1350 + np.log10(config.BOLOMETRIC_CORRECTION)
    Ledd = mass + 38 + np.log10(1.26)
    return Lbol - Ledd


init_guess1 = [ 1, 1549, 30 ]
init_guess2 = [ 1, 1549, 30, 1, 1549, 30 ]
init_guess3 = [ 1, 1549, 30, 1, 1549, 30, 1, 1549, 30 ]

for i, obj_name in enumerate(sorted(os.listdir(config.OUT_FOLDER))):
    obj_name = 'GB6J102623+254255'
    file_path = os.path.join(config.OUT_FOLDER, obj_name)
    print(i, obj_name)

    # Read the spectrum of the object, the astrophysical information and the continuum/mask intervals
    with open(os.path.join(file_path, 'spectrum.pkl'), 'rb') as f:
        obj_spectra = pickle.load(f)
    with open(os.path.join(file_path, 'info.pkl'), 'rb') as f:
        obj_info = pickle.load(f)
    with open(os.path.join(file_path, 'pre_fit.pkl'), 'rb') as f:
        int_dict = pickle.load(f)

    print(int_dict)
    flux = obj_spectra['flux'].values
    lam = obj_spectra['lambda'].values
    ivar = obj_spectra['ivar'].values
    m = int_dict['m']
    q = int_dict['q']
    masks = int_dict['masks']
    cont_lines = int_dict['continuum']

    # trim the spectrum to better help the fitting procedure
    trim = (lam >= 1450) & (lam <= 1650)
    lam = lam[trim]
    flux = flux[trim]
    ivar = ivar[trim]

    # add the default mask 1600-1680 (Denney 2013)
    masks.append(1600)
    masks.append(1680)
    mask_condition = [True for _ in range(len(lam))]
    for i in range(len(masks)//2):
        mask_condition = ((lam < masks[i*2]) | (lam > masks[i*2+1])) & mask_condition

    flux = flux[mask_condition]
    lam = lam[mask_condition]
    ivar = ivar[mask_condition]

    # Continuum subtraction and fit with 1, 2 and 3 gaussians
    flux = flux - (m*lam + q)
    par1, par2, par3 = [], [], []
    try:
        par1, pcov1 = curve_fit(gaussian3, lam, flux, p0=init_guess1, sigma=np.sqrt(1/ivar), maxfev = 500000)
        par2, pcov2 = curve_fit(gaussian3, lam, flux, p0=init_guess2, sigma=np.sqrt(1/ivar), maxfev = 500000)
        par3, pcov3 = curve_fit(gaussian3, lam, flux, p0=init_guess3, sigma=np.sqrt(1/ivar), maxfev = 500000)
    except:
        print(f"{obj_name} Spettro problematico")
        plt.plot(lam, flux)
        plt.savefig(f'problematic_spectra/{obj_name}.png', dpi=300)
        plt.close()
        pass

    if len(par2) == 0: par2=par1
    if len(par3) == 0: par3=par2
    # Plot the 3 differents fits, the user must choose which one better models the spectrum
    fig, axs = plt.subplots(2, 3, figsize=(15,6), sharex=True, sharey=True)
    x_bin = np.arange(int_dict['continuum'][0], int_dict['continuum'][3], 1)
    for ax, par in zip(axs[0,:], [par1, par2, par3]):
        ax.plot(lam, flux, color='black', lw=0.5)
        for i in range(len(masks)//2):
            ax.axvspan(masks[i*2], masks[i*2+1], color='gray', alpha=0.5)
        for line in cont_lines:
            ax.axvline(line, color='green', ls='--')
        ax.plot(x_bin, gaussian3(x_bin, *par), color='red', zorder=10)
    for ax, par in zip(axs[1,:], [par1, par2, par3]):
        ax.plot(lam, flux-gaussian3(lam, *par), color='black', lw=0.5)
        ax.axhline(0, zorder=10)
    fig.tight_layout()
    plt.show()

    n = float(input('Choose number of gaussian (1, 2 or 3):'))

    # The choice is set as the preferred model
    if n == 1:
        par = abs(par1)
        pcov = pcov1
    elif n == 2:
        par = abs(par2)
        pcov = pcov2
    elif n == 3:
        par = abs(par3)
        pcov = pcov3
    else: raise

    # Estimate the best parameters, and save them in a parameters dictionary (pd)
    pdict = {}
    pdict['N_gaussians'] = n
    area, line_disp = calc_line_dispersion(*par)
    y_gaus = gaussian3(x_bin, *par)
    pdict['fwhm'] = calc_fwhm(x_bin, y_gaus)*299792/1549
    pdict['line_disp'] = line_disp*299792/1549

    z = obj_info['z'].values[0]
    dl, _ = utils.ned_calc(z)

    pdict['LCIV'] = flux_to_lum(area, dl)
    f1350 = 1350*m + q
    pdict['lamL1350'] = flux_to_lum(f1350, dl) + np.log10(1350)
    pdict['Lbol'] = pdict['lamL1350'] + np.log10(config.BOLOMETRIC_CORRECTION)
    pdict['Msigma'] = calc_mass_VP06(pdict['line_disp'], pdict['lamL1350'], 'sigma')
    pdict['Mfwhm'] = calc_mass_VP06(pdict['fwhm'], pdict['lamL1350'], 'fwhm')
    pdict['Rsigma'] = calc_edd_ratio(pdict['lamL1350'], pdict['Msigma'])
    pdict['Rfwhm'] = calc_edd_ratio(pdict['lamL1350'], pdict['Mfwhm'])

    # Calculate directly from 'ivar' the uncertainty on the luminosity
    ivar_near_1350 = obj_spectra['ivar'][(obj_spectra['lambda'] >= 1325) & (obj_spectra['lambda'] <= 1335)].values
    if ivar_near_1350.size == 0: ivar_near_1350 = obj_spectra['ivar']
    pdict['lamL1350_err'] = np.log10(1 + np.sqrt(1/np.mean(ivar_near_1350))/f1350)

    # Calculate the uncertainties on the other parameters using a MC method
    p0 = par
    area_mock, line_disp_mock, fwhm_mock = [], [], []
    mock = np.random.normal(flux, np.sqrt(1/ivar), size=(config.N_MONTECARLO, len(flux)))
        # Simulate dead pixels
    dead_pixel_mask = np.random.binomial(1, p=0.95, size=(config.N_MONTECARLO, len(flux)))
    mock = np.multiply(mock, dead_pixel_mask)
    lamL1350_mock = np.random.normal(pdict['lamL1350'], pdict['lamL1350_err'], config.N_MONTECARLO)
    bad_fit = 0
    for i in range(config.N_MONTECARLO):
        try: par_mock, pcov_mock = curve_fit(gaussian3, lam, mock[i], p0=p0, sigma=np.sqrt(1/ivar))
        except:
            bad_fit += 1
            if bad_fit >= config.N_MONTECARLO//4:
                print("Spettro problematico")
                plt.plot(lam, flux)
                plt.savefig(f'problematic_spectra/{obj_name}.png', dpi=300)
                continue
        am, lm = calc_line_dispersion(*par_mock)
        temp = lm*299792/1549
        if (temp/pdict['line_disp'] <= 0.3) | (temp/pdict['line_disp'] >= 3):
            continue
        area_mock.append(am)
        line_disp_mock.append(lm)
        y_gaus_mock = gaussian3(x_bin, *par_mock)
        fwhm_mock.append(calc_fwhm(x_bin, y_gaus_mock))

    f_mock = np.array(fwhm_mock)*(299792/1549)
    l_mock = np.array(line_disp_mock)*(299792/1549)
    plt.hist(f_mock/(pdict['fwhm']), bins=50)
    plt.show()
    plt.hist(l_mock/pdict['line_disp'], bins=50)
    plt.show()

    lamL1350_mock = np.random.normal(pdict['lamL1350'], pdict['lamL1350_err'], len(line_disp_mock))
    Msigma_mock = calc_mass_VP06(np.array(line_disp_mock), np.array(lamL1350_mock), 'sigma')
    Mfwhm_mock = calc_mass_VP06(np.array(fwhm_mock), np.array(lamL1350_mock), 'fwhm')

    pdict['LCIV_err'] = np.std(np.log10(area_mock))
    pdict['Lbol_err'] = pdict['lamL1350_err']
    pdict['fwhm_err'] = np.std(fwhm_mock)*299792/1549
    pdict['line_disp_err'] = np.std(line_disp_mock)*299792/1549
    pdict['Msigma_err'] = np.std(Msigma_mock)
    pdict['Mfwhm_err'] = np.std(Mfwhm_mock)
    pdict['Rsigma_err'] = np.std( calc_edd_ratio(lamL1350_mock, Msigma_mock) )
    pdict['Rfwhm_err'] = np.std( calc_edd_ratio(lamL1350_mock, Mfwhm_mock) )

    print(pdict)
    # Save the dictionary containing the parameters in a pickle file for later
    with open(os.path.join(file_path, 'parameters.pkl'), 'wb') as f:
        pickle.dump(pdict, f)

    # Save the plot with chosen fit
    fig, axs = plt.subplots(2, 1, figsize=(7,7), sharex=True)
    axs[0].plot(lam, flux, color='black', lw=0.5)
    for i in range(len(masks)//2):
        axs[0].axvspan(masks[i*2], masks[i*2+1], color='gray', alpha=0.5)
    for line in cont_lines:
        axs[0].axvline(line, color='green', ls='--')
    axs[0].plot(lam, flux, color='black', lw=0.5)
    axs[0].plot(x_bin, gaussian3(x_bin, *par), color='red', zorder=10)
    axs[0].set_xlabel(r"wavelength ($\AA$)")
    axs[0].set_ylabel(r"flux (10$^{-17}$ erg s$^{-1}$ $\AA^{-1}$)")
    axs[0].set_xlim([1475, 1600])

    axs[1].plot(lam, flux-gaussian3(lam, *par), color='black', lw=0.5)
    axs[1].axhline(0, zorder=10)
    axs[1].set_xlabel(r"wavelength ($\AA$)")
    axs[1].set_ylabel(r"flux (10$^{-17}$ erg s$^{-1}$ $\AA^{-1}$)")

    fig.tight_layout()
    plt.savefig(os.path.join(config.FITTED_PLOTS, obj_name + '.png'), dpi=300)
    plt.close(fig)
