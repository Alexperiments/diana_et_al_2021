import config
import shen_qlf
from astropy.table import Table
from extinction import fitzpatrick99 as fp99
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
import seaborn as sns
import os
import pickle


def rad_opt_limits(zz, z_ref=4, mag_ref=20.86, rad_ref=30):
    '''Limiti ottico e radio per ottenere oggetti con la stessa luminosità
    minima di un oggetto a z=4 con mag-r = 21 e flusso radio a 5GHz = 30 mJy.
    La magnitudine di riferimento non corretta per l'estinzione è 21, con
    la extinction corr. media del campione z>4 in banda r (0.14) si ottiene
    una magnitudine di 20.86.'''
    dist_z, _ = ned_calc(zz)
    dist_ref, _ = ned_calc(z_ref)
    dist_ratio = dist_ref / dist_z
    k_coeff_o = config.ALPHA_O - 1
    k_coeff_r = config.ALPHA_R - 1
    radio_limit = rad_ref * dist_ratio ** 2 * ((1 + z_ref) / (1 + zz)) ** k_coeff_r
    mag_limit = mag_ref - 5 * np.log10(dist_ratio) - 2.5 * k_coeff_o * np.log10((1 + z_ref) / (1 + zz))
    return radio_limit, mag_limit


def ned_calc(z, H0=70, Omega_m=0.3, Omega_vac=0.7):
    '''Script basato sul NED cosmology calculator, per stimare DL e V(Gpc)'''
    # initialize constants
    WM = Omega_m  # Omega(matter)
    WV = Omega_vac  # Omega(vacuum) or lambda
    WR = 0.  # Omega(radiation)
    WK = 0.  # Omega curvaturve = 1-Omega(total)
    c = 299792.458  # velocity of light in km/sec
    Tyr = 977.8  # coefficent for converting 1/H into Gyr
    DTT = 0.5  # time from z to now in units of 1/H0
    DTT_Gyr = 0.0  # value of DTT in Gyr
    age = 0.5  # age of Universe in units of 1/H0
    age_Gyr = 0.0  # value of age in Gyr
    zage = 0.1  # age of Universe at redshift z in units of 1/H0
    zage_Gyr = 0.0  # value of zage in Gyr
    DCMR = 0.0  # comoving radial distance in units of c/H0
    DCMR_Mpc = 0.0
    DCMR_Gyr = 0.0
    DA = 0.0  # angular size distance
    DA_Mpc = 0.0
    DA_Gyr = 0.0
    kpc_DA = 0.0
    DL = 0.0  # luminosity distance
    DL_Mpc = 0.0
    DL_Gyr = 0.0  # DL in units of billions of light years
    V_Gpc = 0.0
    a = 1.0  # 1/(1+z), the scale factor of the Universe
    az = 0.5  # 1/(1+z(object))
    h = H0 / 100.
    WR = 4.165E-5 / (h * h)  # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1 - WM - WR - WV
    az = 1.0 / (1 + 1.0 * z)
    age = 0.

    n = 1000  # number of points in integrals
    for i in range(n):
        a = az * (i + 0.5) / n
        adot = np.sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        age = age + 1. / adot
    zage = az * age / n
    zage_Gyr = (Tyr / H0) * zage
    DTT = 0.0
    DCMR = 0.0

    # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    for i in range(n):
        a = az + (1 - az) * (i + 0.5) / n
        adot = np.sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        DTT = DTT + 1. / adot
        DCMR = DCMR + 1. / (a * adot)
    DTT = (1. - az) * DTT / n
    DCMR = (1. - az) * DCMR / n
    age = DTT + zage
    age_Gyr = age * (Tyr / H0)
    DTT_Gyr = (Tyr / H0) * DTT
    DCMR_Gyr = (Tyr / H0) * DCMR
    DCMR_Mpc = (c / H0) * DCMR

    # tangential comoving distance
    ratio = 1.00
    x = np.sqrt(abs(WK)) * DCMR
    if WK > 0:
        ratio = 0.5 * (np.exp(x) - np.exp(-x)) / x
    else:
        ratio = np.sin(x) / x
    DCMT = ratio * DCMR
    DA = az * DCMT
    DA_Mpc = (c / H0) * DA
    kpc_DA = DA_Mpc / 206.264806
    DA_Gyr = (Tyr / H0) * DA
    DL = DA / (az * az)
    DL_Mpc = (c / H0) * DL
    DL_Gyr = (Tyr / H0) * DL

    # comoving volume computation
    ratio = 1.00
    x = np.sqrt(abs(WK)) * DCMR
    if WK > 0:
        ratio = (0.125 * (np.exp(2. * x) - np.exp(-2. * x)) - x / 2.) / (x * x * x / 3.)
    else:
        ratio = (x / 2. - np.sin(2. * x) / 4.) / (x * x * x / 3.)
    VCM = ratio * DCMR * DCMR * DCMR / 3.
    V_Gpc = 4. * np.pi * ((0.001 * c / H0) ** 3) * VCM
    return DL_Mpc, V_Gpc


def gaussian3(x, A, mean1, sigma1, B=0, mean2=0, sigma2=1, C=0, mean3=0, sigma3=1):
    return (A * np.exp(-np.power((x - mean1) / sigma1, 2.) / 2.) +
            B * np.exp(-np.power((x - mean2) / sigma2, 2.) / 2.) +
            C * np.exp(-np.power((x - mean3) / sigma3, 2.) / 2.))


def selection():
    '''Scansiona gli oggetti della CLASS presenti nell'area SDSS (con mag-r e
    |b"| >= 20) e seleziona quelli che rispettano i limiti di luminosità del
    nostro sample e compresi tra 1.5<=z<4.
    A questi aggiunge poi gli oggetti del campione C19.
    Calcola infine il numero di oggetti e quanti di questi non hanno uno spettro
    (nella SDSS, per gli oggetti z<4, o in generale uno spettro elettronico per
    il campione C19).
    Save the selection in a file data/selection.txt
    '''
    class_data = pd.read_csv(config.SDSS_IN_CLASS_FILE, sep='\s+')

    lam = np.array([6231])
    def correct_mag_extinction(mag, av):
        return mag - fp99(lam, av, 3.1)

    class_data['psfmagr_corr'] = np.vectorize(correct_mag_extinction)(class_data['psfmagr'], class_data['a_v'])

    names_with_spectra = os.listdir(config.SOURCE_FOLDER)
    names_with_spectra = [n.strip('.fits').strip('.txt') for n in names_with_spectra]

    C19_names = pd.read_csv("data/C19.txt", sep='\t')['name'].values

    rad_lim, mag_lim = rad_opt_limits(class_data['z'])

    selection_mask = (
            (class_data['psfmagr_corr'] <= mag_lim) &
            (class_data['flag_class'] == 1) &
            (class_data['z'] >= 1.5) &
            (class_data['z'] < 4) &
            (class_data['gbflux'] >= rad_lim)
    )

    select = class_data[selection_mask]

    print(f"Selected objects z<4: {select.shape[0]}")

    without_spectra = [n for n in select['classname'] if n not in names_with_spectra]

    print(f"z<4 objects lacking a SDSS spectrum: {len(without_spectra)}")

    C19 = []
    for n in C19_names:
        df = class_data[class_data['classname'] == n]
        C19.append(df)

    C19.append(pd.DataFrame(
        {
            'classname': ['GB6J164856+460341', 'GB6J090631+693027'],
            'gbflux': [36, 114],
            'psfmagr': [20.31, 20.54],
            'flag_class': [1, 1],
            'z': [5.36000, 5.47000],
            'RA': [0.00000, 0.00000],
            'DEC': [0.00000, 0.00000],
            'bii': [0.00000, 0.00000],
            'lii': [0.00000, 0.00000],
            'a_v': [0.00000, 0.00000]
        }
    ))

    C19 = pd.concat(C19)

    total = pd.concat([select, C19])

    print(f"Selected objects: {total.shape[0]}")

    without_spectra = [n for n in total['classname'] if n not in names_with_spectra]

    print(f"Lacking a spectrum: {len(without_spectra)}")

    total.to_csv(config.SELECTION_FILE, sep='\t', index=False)


def save_parameters_list():
    '''I parametri fisici calcolati con il fit sono salvati in file
    separati per ciascun oggetto. Questo script raccoglie le varie
    informazioni e le salva in un unico file per un comodo accesso.
    Aggiunge manualmente i parametri dei due oggetti con spettro
    cartaceo.
    '''
    total_df = pd.DataFrame()
    for i, obj_name in enumerate(sorted(os.listdir(config.OUT_FOLDER))):
        file_path = os.path.join(config.OUT_FOLDER, obj_name)

        with open(os.path.join(file_path, 'info.pkl'), 'rb') as f:
            obj_info = pickle.load(f)
        with open(os.path.join(file_path, 'parameters.pkl'), 'rb') as f:
            parameters = pickle.load(f)

        series = obj_info[['classname', 'z', 'psfmagr', 'gbflux']]
        param_df = pd.DataFrame(parameters, index=[0])
        series = pd.concat([series, param_df], axis=1)
        total_df = pd.concat([total_df, series], axis=0)

    all_selected = pd.read_csv(config.SELECTION_FILE, sep='\t')
    all_selected = all_selected[['classname', 'gbflux', 'psfmagr', 'z']]

    total_df = total_df.merge(all_selected, how='outer', on=['classname', 'gbflux', 'psfmagr', 'z'])

    total_df = total_df[total_df['classname'] != 'GB6J090631+693027']
    total_df = total_df[total_df['classname'] != 'GB6J012202+030951']

    # Add the two objects without a digital spectra
    obj1 = pd.DataFrame({
        'classname': 'GB6J012202+030951',
        'z': 4.00,
        'psfmagr': 20.86,
        'gbflux': 96,
        'fwhm': 5800,
        'fwhm_err': 2000,
        'lamL1350': 46.450,
        'lamL1350_err': 0.070,
        'Lbol': 47.103,
        'Lbol_err': 0.070,
        'Mfwhm': 9.52,
        'Mfwhm_err': 0.39,
        'Msigma': 9.52,
        'Msigma_err': 0.39,
        'Rfwhm': -0.51,
        'Rfwhm_err': 0.39
    }, index=[378])
    obj2 = pd.DataFrame({
        'classname': 'GB6J090631+693027',
        'z': 5.47,
        'psfmagr': 20.54,
        'gbflux': 100,
        'Msigma': 9.30,
        'Msigma_err': 0.39,
        'Mfwhm': 9.30,
        'Mfwhm_err': 0.39,
    }, index=[379])

    total_df = pd.concat([total_df, obj1, obj2], axis=0)
    total_df = total_df.sort_values(by='classname')

    with open(config.PARAMETERS_FILE, 'wb') as f:
        pickle.dump(total_df, f)


def stamp_plot_diskVSse():
    '''Produce il plot che confronta le masse calcolate con l'AD e
    quelle calcolate con il SE. Il confronto è fatto solo sugli oggetti
    per cui abbiamo le masse da disco (z>4).
    Stampa la differenza logaritmica delle due distribuzioni con dev. std.
    al momento la stima è 0.11+-0.31 compatibile con uno scatter di 0.4dex
    (non 0.3dex, in quanto gli oggetti a z>4 sono prevalentemente con basso
    S/N).
    '''
    with open(config.PARAMETERS_FILE, 'rb') as f:
        parameters = pickle.load(f)

    with open("data/disk_masses.txt", "r") as fp:
        lines = fp.readlines()[1:]
        names = [line.split()[0] for line in lines]
        dmass = [float(line.split()[2]) for line in lines]
        massup = [float(line.split()[3]) - float(line.split()[2]) for line in lines]
        massdown = [float(line.split()[4]) - float(line.split()[2]) for line in lines]

    diff = []
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    for n, diskmass, d, u in zip(names, dmass, massdown, massup):
        SEmass = parameters[parameters['classname'] == n]
        ax.errorbar(SEmass['Msigma'], diskmass, xerr=SEmass['Msigma_err'], yerr=[[-d], [u]], fmt='.k', elinewidth=0.7,
                    markersize=5, capsize=2, capthick=0.7, zorder=10)
        diff.append(SEmass['Msigma'] - diskmass)

    print("### SE mass - disk mass ###")
    print(f"ratio: {np.mean(diff):.2f} +- {np.std(diff):.2f}")

    bisec = np.arange(8, 10.1, 0.1)
    ax.plot(bisec, bisec, lw=1, color='#dd4c65', zorder=0)
    ax.fill_between(bisec, bisec + 0.4, bisec - 0.4, facecolor='#93c4d2', zorder=0)
    ax.set_xlim([8, 10])
    ax.set_ylim([8, 10])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlabel(r"LogM$_{\sigma}$ (M$_{\odot}$)")
    ax.set_ylabel(r"LogM$_{disk}$ (M$_{\odot}$)")
    plt.savefig(config.PAPER_PLOTS + "diskVsVirial.pdf")


def stamp_plot_LCIV_1350():
    '''Produce il plot che confronta le distribuzioni di LCIV e L1350
    del nostro campione e del campione Shen. Stampa la media e la dev.
    std. della differenza logaritmica tra LCIV-L1350.
    '''
    dat = Table.read('data/dr7_bh_Nov19_2013.fits', format='fits')
    names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
    df = dat[names].to_pandas()
    mask = ((df['LOGL1350'] == 0) | (df['LOGL_CIV'] == 0))
    df.mask(mask, inplace=True, axis=0)
    df.dropna(axis=0, how='any', inplace=True)

    x_shen = df['LOGL1350']
    y_shen = df['LOGL_CIV']

    with open(config.PARAMETERS_FILE, 'rb') as f:
        parameters = pickle.load(f)
    x_our = parameters['lamL1350']
    y_our = parameters['LCIV']

    ratio_shen = x_shen - y_shen
    ratio_our = x_our - y_our

    mean_shen = np.mean(ratio_shen)
    std_shen = np.std(ratio_shen)
    mean_our = np.mean(ratio_our)
    std_our = np.std(ratio_our)
    print("### Shen sample ###")
    print(f"ratio: {mean_shen:.2f} +- {std_shen:.2f}")
    print("### Class sample ###")
    print(f"ratio: {mean_our:.2f} +- {std_our:.2f}")

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    x_fit = np.arange(44.5, 47.5, 0.1)
    sns.set_style("white")
    sns.kdeplot(x_shen, y_shen, cmap="Blues", shade=True, thresh=0, zorder=-1, n_levels=10)
    ax.scatter(x_our, y_our, s=3, color='black', zorder=1)
    plt.xlabel(r"Log$\lambda$L$_{1350}$ (erg s$^{-1}$)")
    plt.ylabel(r"LogL$_{CIV}$ (erg s$^{-1}$)")
    plt.xlim([44.5, 47.5])
    plt.ylim([43, 46])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(config.PAPER_PLOTS + "LCIV-L1350.eps")


def merge_columns_with_uncertainty(df):
    df.reset_index(inplace=True)
    columns = df.columns
    columns_noerr = set(n.replace('_err', "") for n in columns)
    columns_need = [n.replace('_err', "") for n in columns if n not in columns_noerr]
    df[columns_need] = df[columns_need].astype(str)
    for i, series in df.iterrows():
        for c in columns_need:
            value = (f"${series[c]}" + r'\pm' + f"{series[c + '_err']}$").replace('.0\\pm', '\\pm').replace('.0$',
                                                                                                            '$').replace(
                '$nan\pmnan$', '-')
            df.loc[i, c] = value
    return df


def print_C19_table():
    C19_names = [
        'GB6J001115+144608',
        'GB6J012202+030951',
        'GB6J083548+182519',
        'GB6J083945+511206',
        'GB6J090631+693027',
        'GB6J091825+063722',
        'GB6J102107+220904',
        'GB6J102623+254255',
        'GB6J132512+112338',
        'GB6J134811+193520',
        'GB6J141212+062408',
        'GB6J143023+420450',
        'GB6J151002+570256',
        'GB6J153533+025419',
        'GB6J162956+095959',
        'GB6J164856+460341',
        'GB6J171103+383016',
        'GB6J231449+020146',
        'GB6J235758+140205',
    ]
    with open(config.PARAMETERS_FILE, 'rb') as f:
        parameters = pickle.load(f)

    C19_df = [parameters[parameters['classname'] == n] for n in C19_names]
    C19_df = pd.concat(C19_df)
    selection_round2 = [
        'z',
        'psfmagr',
        'Msigma',
        'Msigma_err',
        'Mfwhm',
        'Mfwhm_err',
        'Rsigma',
        'Rsigma_err',
        'Rfwhm',
        'Rfwhm_err'
    ]
    selection_round0 = [
        'gbflux',
        'line_disp',
        'fwhm',
        'line_disp_err',
        'fwhm_err'
    ]
    selection_round3 = [
        'LCIV',
        'LCIV_err',
        'lamL1350',
        'lamL1350_err',
        'Lbol',
        'Lbol_err'
    ]
    selection_toprint = [
        'classname',
        'z',
        'psfmagr',
        'gbflux',
        'line_disp',
        'fwhm',
        'lamL1350',
        'LCIV',
        'Lbol',
        'Msigma',
        'Mfwhm',
        'Rsigma',
        'Rfwhm'
    ]

    idx = C19_df.index[C19_df['classname'] == 'GB6J012202+030951']
    C19_df.loc.__setitem__((idx, 'Msigma'), np.nan)
    C19_df.loc.__setitem__((idx, 'Msigma_err'), np.nan)

    for name, values in C19_df.iteritems():
        if name in selection_round2:
            C19_df[name] = C19_df[name].map('{:.2f}'.format)
        elif name in selection_round3:
            C19_df[name] = C19_df[name].map('{:.3f}'.format)
        elif name in selection_round0:
            C19_df[name] = C19_df[name].map('{:.0f}'.format)

    C19_df = merge_columns_with_uncertainty(C19_df)
    C19_df.to_latex(buf=config.C19_TEX, columns=selection_toprint,
                    na_rep='-', index=False, header=False, escape=False,
                    )


def flux_to_lum(f, z, alpha):
    dl, _ = ned_calc(z)
    k_corr = (alpha - 1) * np.log10(1 + z)
    return f + np.log10(4 * math.pi) + 2 * np.log10(3.08 * dl) + 48 + k_corr


def get_luminosity_limits():
    z_ref = 4
    dl, _ = ned_calc(z_ref)  # Lum. distance in Mpc
    sph_area = np.log10(4 * math.pi) + 2 * np.log10(3.08 * dl) + 48  # sphere area in cm

    rflux_mjy, ab_mag = rad_opt_limits(z_ref)
    rflux = np.log10(rflux_mjy) - 33  # radio flux (function of frequency) in W cm-2 Hz-1
    # invariant optical flux (function of wavelength) at 6231A
    lam_oflux = np.log10(299792458) + 10 - np.log10(6231) - (ab_mag + 48.6) / 2.5
    nu_oflux = (ab_mag + 48.6) / 2.5

    k_corr_o = (config.ALPHA_O - 1) * np.log10(1 + z_ref)
    lam_olum = lam_oflux + sph_area + k_corr_o + (config.ALPHA_O - 1) * np.log10(1350 / 6231)

    k_corr_r = (config.ALPHA_R - 1) * np.log10(1 + z_ref)
    rlum = rflux + sph_area + k_corr_r

    min_L_eff = shen_qlf.calc_lum_eff()

    Lbol = lam_olum + np.log10(config.BOLOMETRIC_CORRECTION)
    Ledd = 9 + 38 + np.log10(1.26)
    Redd = Lbol - Ledd

    print(f"Optical invariant luminosity cut at 1350A: {lam_olum:.2f} erg s-1")
    print(f"Radio power cut at 5GHz: {rlum:.2f} W Hz-1")
    print(f"Minimum luminosity to integrate Shen QLF: {min_L_eff:.2f} erg s-1")
    print(f"Minimum Eddington luminosity: {10 ** Redd:.2f}")


def print_example_spectrum():
    obj_name = 'GB6J162030+490149'
    file_path = os.path.join(config.OUT_FOLDER, obj_name)

    # Read the spectrum of the object, the astrophysical information and the continuum/mask intervals
    with open(os.path.join(file_path, 'spectrum.pkl'), 'rb') as f:
        obj_spectra = pickle.load(f)
    with open(os.path.join(file_path, 'pre_fit.pkl'), 'rb') as f:
        int_dict = pickle.load(f)
    with open(os.path.join(file_path, 'parameters.pkl'), 'rb') as f:
        parameters = pickle.load(f)

    flux = obj_spectra['flux']
    lam = obj_spectra['lambda']
    ivar = obj_spectra['ivar']
    N_gaus = parameters['N_gaussians']
    continuum = int_dict['continuum']
    masks = int_dict['masks']
    m = int_dict['m']
    q = int_dict['q']

    trim = (lam >= 1450) & (lam <= 1650)
    lam = lam[trim]
    flux = flux[trim]
    ivar = ivar[trim]

    masks.append(1600)
    masks.append(1680)
    mask_condition = [True for _ in range(len(lam))]
    for i in range(len(masks) // 2):
        mask_condition = ((lam < masks[i * 2]) | (lam > masks[i * 2 + 1])) & mask_condition

    flux = flux - (m * lam + q)
    mask_flux = flux[mask_condition]
    mask_lam = lam[mask_condition]
    mask_ivar = ivar[mask_condition]

    par, _ = curve_fit(
        gaussian3, mask_lam, mask_flux,
        p0=[
            1, 1549, 30, 1, 1549, 30, 1, 1549, 30
        ],
        sigma=np.sqrt(1 / mask_ivar)
    )

    fig = plt.figure(figsize=(5.5, 5.35))
    grid = plt.GridSpec(7, 1)
    grid.update(wspace=0., hspace=0.)
    ax = plt.subplot(grid[0:6])
    ax2 = plt.subplot(grid[6:7], sharex=ax)
    ax2.set_xticks([1500, 1600])
    ax2.set_yticks([0])
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.get_shared_x_axes().join(ax, ax2)
    x_bin = np.arange(continuum[0], continuum[3], 1)
    ax.plot(mask_lam, mask_flux, color='black', lw=0.75, zorder=5)
    ax.plot(lam, flux, color='orange', lw=0.5)
    for axes in [ax, ax2]:
        for i in range(len(masks) // 2):
            axes.axvspan(masks[i * 2], masks[i * 2 + 1], color='gray', zorder=0, alpha=0.5)
        # for line in continuum:
        #    axes.axvline(line, color='green', ls='--')
    ax.plot(x_bin, gaussian3(x_bin, *par), color='red', zorder=10)
    ax.plot(x_bin, gaussian3(x_bin, *par[0:3]), color='red', ls='dotted', zorder=10)
    ax.plot(x_bin, gaussian3(x_bin, *par[3:6]), color='red', ls='dotted', zorder=10)
    ax.plot(x_bin, gaussian3(x_bin, *par[6:9]), color='red', ls='dotted', zorder=10)
    ax.axhline(0, zorder=5, ls='--', color='blue')
    ax.set_xlim([1450, 1650])
    ax.set_ylim([0, 100])
    ax2.plot(mask_lam, mask_flux - gaussian3(mask_lam, *par), color='black', lw=0.5)
    ax2.axhline(0, zorder=10, ls='--', color='blue')
    ax2.set_xlabel(r"wavelength ($\AA$)")
    ax.set_ylabel(r"flux (10$^{-17}$ erg s$^{-1}$ $\AA^{-1}$)")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(config.PAPER_PLOTS + "example_plot.pdf")


def get_other_works_limits(author, lam_opt_filter, radio_freq, mag_lim, radio_lim, z_ref=4):
    zz = 4
    dl, _ = ned_calc(zz)  # Lum. distance in Mpc
    sph_area = np.log10(4 * math.pi) + 2 * np.log10(3.08 * dl) + 48  # sphere area in cm
    rflux_mjy, ab_mag = rad_opt_limits(zz, z_ref=z_ref, mag_ref=mag_lim, rad_ref=radio_lim)

    rflux = np.log10(rflux_mjy) - 33  # radio flux (function of frequency) in W cm-2 Hz-1
    # invariant optical flux (function of wavelength) at 6231A
    lam_oflux = np.log10(299792458) + 10 - np.log10(lam_opt_filter) - (ab_mag + 48.6) / 2.5
    nu_oflux = (ab_mag + 48.6) / 2.5

    k_corr_o = (config.ALPHA_O - 1) * np.log10(1 + zz)
    lam_olum = lam_oflux + sph_area + k_corr_o + (config.ALPHA_O - 1) * np.log10(1350 / lam_opt_filter)

    k_corr_r = (config.ALPHA_R - 1) * np.log10(1 + zz)
    rlum = rflux + sph_area + k_corr_r

    print(f"Optical limit {author} at 1350A: {lam_olum:.2f} erg s-1")
    print(f"Radio limit {author} at {radio_freq:.1f}GHz: {rlum:.2f} W Hz-1")


def calculate_radioloud_lim(gamma=10, boost_factor=2.5):
    z_ref = 4
    dl, _ = ned_calc(z_ref)  # Lum. distance in Mpc
    sph_area = np.log10(4 * math.pi) + 2 * np.log10(3.08 * dl) + 48  # sphere area in cm

    rflux_mjy, ab_mag = rad_opt_limits(z_ref)
    rflux = np.log10(rflux_mjy) - 26  # radio flux (function of frequency) in W cm-2 Hz-1
    # invariant optical flux (function of wavelength) at 6231A
    ofluxnu = - (ab_mag + 48.6) / 2.5

    k_corr_o = (config.ALPHA_O - 1) * np.log10(1 + z_ref)
    f4400_nu = ofluxnu + k_corr_o + (-config.ALPHA_O) * np.log10(4400 / 6231)

    k_corr_r = (config.ALPHA_R - 1) * np.log10(1 + z_ref)
    f5ghz = rflux + k_corr_r
    deboosted_f5ghz = f5ghz - np.log10(gamma ** boost_factor)

    print(f"Minimum Radioloudness sensitivity: {deboosted_f5ghz - f4400_nu}")


def print_allsample_table():
    with open(config.PARAMETERS_FILE, 'rb') as f:
        all = pickle.load(f)
    selection_round2 = [
        'z',
        'psfmagr',
        'Msigma',
        'Msigma_err',
        'Mfwhm',
        'Mfwhm_err',
        'Rsigma',
        'Rsigma_err',
        'Rfwhm',
        'Rfwhm_err'
    ]
    selection_round0 = [
        'gbflux',
        'line_disp',
        'fwhm',
        'line_disp_err',
        'fwhm_err'
    ]
    selection_round3 = [
        'LCIV',
        'LCIV_err',
        'lamL1350',
        'lamL1350_err',
        'Lbol',
        'Lbol_err'
    ]
    selection_toprint = [
        'classname',
        'z',
        'psfmagr',
        'gbflux',
        'line_disp',
        'fwhm',
        'lamL1350',
        'LCIV',
        'Lbol',
        'Msigma',
        'Mfwhm',
        'Rsigma',
        'Rfwhm'
    ]

    idx = all.index[all['classname'] == 'GB6J012202+030951']
    all.loc.__setitem__((idx, 'Msigma'), np.nan)
    all.loc.__setitem__((idx, 'Msigma_err'), np.nan)

    for name, values in all.iteritems():
        if name in selection_round2:
            all[name] = all[name].map('{:.2f}'.format)
        elif name in selection_round3:
            all[name] = all[name].map('{:.3f}'.format)
        elif name in selection_round0:
            all[name] = all[name].map('{:.0f}'.format)

    all = merge_columns_with_uncertainty(all)
    all.to_latex(buf=config.TABLE_TEX, columns=selection_toprint,
                 na_rep='-', index=False, header=False, escape=False,
                 )


def main():
    # config.default_plot_settings()
    selection()
    # save_parameters_list()
    # stamp_plot_diskVSse()
    # stamp_plot_LCIV_1350()
    # print_C19_table()
    # print_allsample_table()
    # get_luminosity_limits()
    # get_other_works_limits('Banados', 6170, 1.4, 22.8, 1)
    # get_other_works_limits('Kratzer & Richards Sample B', 7630, 1.4, 20.1, 1, 3)
    # calculate_radioloud_lim(7, 3)
    # print_example_spectrum()


if __name__ == '__main__':
    main()
