import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

def rad_opt_limits(zz, z_ref=4, mag_ref=21, rad_ref=30, beta=False):
    dist_z, _ = ned_calc(zz)
    dist_ref, _ = ned_calc(z_ref)
    dist_ratio = dist_ref/dist_z
    k_coeff_o = config.ALPHA_O-1
    k_coeff_r = config.ALPHA_R-1
    radio_limit = rad_ref*dist_ratio**2*((1+z_ref)/(1+zz))**k_coeff_r
    mag_limit = mag_ref-5*np.log10(dist_ratio)-2.5*k_coeff_o*np.log10((1+z_ref)/(1+zz))
    if beta: mag_limit = mag_limit - 2.5*config.ALPHA_O*np.log10(1400/6231)
    return radio_limit,mag_limit


def ned_calc(z, H0=70, Omega_m=0.3, Omega_vac=0.7):
    # initialize constants
    WM = Omega_m   # Omega(matter)
    WV = Omega_vac # Omega(vacuum) or lambda
    WR = 0.        # Omega(radiation)
    WK = 0.        # Omega curvaturve = 1-Omega(total)
    c = 299792.458 # velocity of light in km/sec
    Tyr = 977.8    # coefficent for converting 1/H into Gyr
    DTT = 0.5      # time from z to now in units of 1/H0
    DTT_Gyr = 0.0  # value of DTT in Gyr
    age = 0.5      # age of Universe in units of 1/H0
    age_Gyr = 0.0  # value of age in Gyr
    zage = 0.1     # age of Universe at redshift z in units of 1/H0
    zage_Gyr = 0.0 # value of zage in Gyr
    DCMR = 0.0     # comoving radial distance in units of c/H0
    DCMR_Mpc = 0.0
    DCMR_Gyr = 0.0
    DA = 0.0       # angular size distance
    DA_Mpc = 0.0
    DA_Gyr = 0.0
    kpc_DA = 0.0
    DL = 0.0       # luminosity distance
    DL_Mpc = 0.0
    DL_Gyr = 0.0   # DL in units of billions of light years
    V_Gpc = 0.0
    a = 1.0        # 1/(1+z), the scale factor of the Universe
    az = 0.5       # 1/(1+z(object))
    h = H0/100.
    WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1-WM-WR-WV
    az = 1.0/(1+1.0*z)
    age = 0.

    n=1000         # number of points in integrals
    for i in range(n):
        a = az*(i+0.5)/n
        adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        age = age + 1./adot
    zage = az*age/n
    zage_Gyr = (Tyr/H0)*zage
    DTT = 0.0
    DCMR = 0.0

    # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    for i in range(n):
        a = az+(1-az)*(i+0.5)/n
        adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        DTT = DTT + 1./adot
        DCMR = DCMR + 1./(a*adot)
    DTT = (1.-az)*DTT/n
    DCMR = (1.-az)*DCMR/n
    age = DTT+zage
    age_Gyr = age*(Tyr/H0)
    DTT_Gyr = (Tyr/H0)*DTT
    DCMR_Gyr = (Tyr/H0)*DCMR
    DCMR_Mpc = (c/H0)*DCMR

    # tangential comoving distance
    ratio = 1.00
    x = np.sqrt(abs(WK))*DCMR
    if WK > 0:
      ratio =  0.5*(np.exp(x)-np.exp(-x))/x
    else:
      ratio = np.sin(x)/x
    DCMT = ratio*DCMR
    DA = az*DCMT
    DA_Mpc = (c/H0)*DA
    kpc_DA = DA_Mpc/206.264806
    DA_Gyr = (Tyr/H0)*DA
    DL = DA/(az*az)
    DL_Mpc = (c/H0)*DL
    DL_Gyr = (Tyr/H0)*DL

    # comoving volume computation
    ratio = 1.00
    x = np.sqrt(abs(WK))*DCMR
    if WK > 0:
      ratio = (0.125*(np.exp(2.*x)-np.exp(-2.*x))-x/2.)/(x*x*x/3.)
    else:
      ratio = (x/2. - np.sin(2.*x)/4.)/(x*x*x/3.)
    VCM = ratio*DCMR*DCMR*DCMR/3.
    V_Gpc = 4.*np.pi*((0.001*c/H0)**3)*VCM
    return DL_Mpc, V_Gpc


def selection():
    class_data = pd.read_csv(config.SDSS_IN_CLASS_FILE, sep='\t')

    names_with_spectra = os.listdir(config.HDF_FOLDER)

    C19_data = pd.read_csv("data/C19.txt", sep='\t')

    rad_lim, mag_lim = rad_opt_limits(class_data['z'], z_ref=4.5, beta=True)

    selection_mask1 = (
        (class_data['psfmagr'] <= mag_lim)  &
        (class_data['flag_class'] == 1)     &
        (class_data['z'] >= 1.5)            &
        (class_data['z'] < 4)               &
        (class_data['gbflux'] >= rad_lim)
    )

    rad_lim, mag_lim = rad_opt_limits(class_data['z'], z_ref=4, beta=False)

    selection_mask2 = (
        (class_data['psfmagr'] <= mag_lim)  &
        (class_data['flag_class'] == 1)     &
        (class_data['z'] >= 1.5)            &
        (class_data['z'] < 4)               &
        (class_data['gbflux'] >= rad_lim)
    )

    old_select = class_data[selection_mask1]
    new_select = class_data[selection_mask2]

    print(f"Selected objects old: {old_select.shape[0]}")
    print(f"Selected objects old: {new_select.shape[0]}")

    without_spectra = [print(n) for n in names_with_spectra if n not in old_select['classname'].values]
    [print(n) for n in names_with_spectra if (n not in new_select['classname'].values) and (n not in without_spectra)]

    #select.to_csv(config.SELECTION_FILE, sep='\t', index=False)

    z_bin = np.arange(1.5, 6, 0.1)
    plt.scatter(old_select['z'], old_select['psfmagr'], color='black', marker='^')
    plt.scatter(new_select['z'], new_select['psfmagr'], color='red', marker='v')
    plt.plot(z_bin, rad_opt_limits(z_bin, z_ref=4)[1], color='red')
    plt.plot(z_bin, rad_opt_limits(z_bin, z_ref=4.5, beta=True)[1], color='black')
    plt.show()
    plt.scatter(old_select['z'], old_select['gbflux'], color='black', marker='^')
    plt.scatter(new_select['z'], new_select['gbflux'], color='red', marker='v')
    plt.plot(z_bin, rad_opt_limits(z_bin, z_ref=4)[0], color='red')
    plt.plot(z_bin, rad_opt_limits(z_bin, z_ref=4.5, beta=True)[0], color='black')
    plt.show()
    #+ 2.5*config.ALPHA_O*np.log10(1400/6231)

def test():
    selection()

if __name__ == '__main__':
    test()
