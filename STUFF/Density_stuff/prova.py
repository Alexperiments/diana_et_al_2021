#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics

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
    if x > 0.1:
        if WK > 0:
          ratio =  0.5*(exp(x)-exp(-x))/x
        else:
          ratio = sin(x)/x
    else:
        y = x*x
        if WK < 0: y = -y
        ratio = 1. + y/6. + y*y/120.
    DCMT = ratio*DCMR
    DA = az*DCMT
    DA_Mpc = (c/H0)*DA
    kpc_DA = DA_Mpc/206.264806
    DA_Gyr = (Tyr/H0)*DA
    DL = DA/(az*az)
    DL_Mpc = (c/H0)*DL

    return DL_Mpc

def Lum_bol(z, mag, lambda_obs):
    flux_obs_nu = -(mag + 48.6)/2.5
    flux_obs_lambda = flux_obs_nu + np.log10(299792458) + 10 - 2*np.log10(lambda_obs)
    Dl = ned_calc(z)
    L_lambda_rest = flux_obs_lambda + np.log10(4*math.pi) + 2*np.log10(Dl*3.09) + 48 + np.log10(lambda_obs)
    k_bol = K_bol_calc(L_lambda_rest)
    return L_lambda_rest + np.log10(k_bol)

def M_min(REDD, z, mag):
    logREDD = np.log10(REDD)
    L_limit = Lum_bol(z, mag, 7520)
    return L_limit - conversion - logREDD

def REDD_min(Mmin, z, mag):
    L_limit = Lum_bol(z, mag, 7520)
    return L_limit - conversion - Mmin

def K_bol_calc(L1400):
    return 3.81#7*(10**(L1400-42))**(-0.1)

edd_ratio = []
z  = []
masse = []
masse_trim = []
radio_flux = []
edd_ratio_trim = []
count1 = 0
count2 = 0
countflux = 0

conversion = math.log10(1.26) + 38
L_limit = Lum_bol(4.5, 21.5, 7520)
'''L_limit_u = Lum_bol(2, 22.0, 3540)
L_limit_g = Lum_bol(2, 22.2, 4750)
L_limit_r = Lum_bol(2, 22.2, 6220)'''

print(10**REDD_min(9, 4.5, 21.5))

with open("sdss_in_class1.txt", 'r') as fp:
    lines = fp.readlines()[1:]
    for line in lines:
        if (float(line.split()[8]) != 0.0) and (float(line.split()[8]) > -100) and (float(line.split()[4]) > 0) and (float(line.split()[4]) < 6) and (float(line.split()[4]) > 0) and (float(line.split()[6]) > 0):
            radio_flux.append(float(line.split()[1]))
            edd_ratio.append(float(line.split()[8]))
            z.append(float(line.split()[4]))
            masse.append(float(line.split()[6]))
        if (float(line.split()[8]) != 0.0) and (float(line.split()[8]) > -100) and (float(line.split()[4]) > 0) and (float(line.split()[4]) < 6) and (float(line.split()[4]) > 1) and (float(line.split()[6]) > 0):
            edd_ratio_trim.append(float(line.split()[8]))
            masse_trim.append(float(line.split()[6]))

#corregge gli REDD per adattarli a posteriori alla correzione bolometrica calcolata da Netzer 2019
L_1350_shen = np.asarray(edd_ratio) + np.asarray(masse) + conversion - math.log10(3.81)
L_1350_sehn_trim = np.asarray(edd_ratio_trim) + np.asarray(masse_trim) + conversion - math.log10(3.81)
edd_ratio_corr = np.asarray(edd_ratio) - math.log10(3.81) + np.log10(K_bol_calc(L_1350_shen))
edd_ratio_trim_corr = np.asarray(edd_ratio_trim) - math.log10(3.81) + np.log10(K_bol_calc(L_1350_sehn_trim))

for i in range(len(edd_ratio)):
    if (edd_ratio_corr[i]) < np.log10(0.22):
        count1 = count1 + 1

for i in range(len(edd_ratio_trim)):
    if (edd_ratio_trim_corr[i] + masse_trim[i] + conversion) < L_limit:
        count2 = count2 + 1

dist4_5 = ned_calc(4.5)
k_corr = 0.44-1
for i in range(len(radio_flux)):

    dist_z = ned_calc(z[i])
    dist_ratio = dist4_5/dist_z
    if radio_flux[i] < 30*dist_ratio**2*((1+4.5)/(1+z[i]))**k_corr:
        countflux = countflux+1
z_bin = np.linspace(0.5,6,100)
dist_ratio = np.zeros(len(z_bin))
for i in range(len(z_bin)):
    dist_z = ned_calc(z_bin[i])
    dist_ratio[i] = dist4_5/dist_z
radio_bin = 30*dist_ratio**2*((1+4.5)/(1+z_bin))**k_corr

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(z, radio_flux, label=r'$S_{5GHz}$', s=0.1)
ax.plot(z_bin, radio_bin, c='black', ls='--', alpha=0.8, label=r'$S_{5GHz}=30mJy$', lw=0.5)
plt.axhline(y=30, ls='--', c='r', lw=0.5)
plt.title(r"$S_{{limit}}$ = {0}; completeness = {1:.0f}%".format("30mJy", 100*(1-countflux/len(radio_flux))))
plt.xlim([0.5,6])
plt.ylim([0,1000])
plt.savefig("RadioCut.png", dpi=300)

'''fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist( edd_ratio_trim_corr + masse_trim + conversion, bins=25, alpha=0.5, label=r'$z\in [0,3]$')
#ax.hist( edd_ratio_trim_corr + np.asarray(masse_trim) + conversion, bins=25, alpha=0.5, label=r'$z\in [1,2]$', color='purple')
#ax.axvline(x=np.log10(0.22), c='black', ls='--', alpha=0.8, label=r'$L_{lim}(z=4.5, i=21.5)$')
#ax.axvline(x=L_limit_u, c='yellow', ls='--', alpha=0.8, label=r'$L_{lim}(z=2, u=22.0)$')
#ax.axvline(x=L_limit_g, c='green', ls='--', alpha=0.8, label=r'$L_{lim}(z=2, g=22.2)$')
#ax.axvline(x=L_limit_r, c='red', ls='--', alpha=0.8, label=r'$L_{lim}(z=2, r=22.2)$')
ax.legend(loc='upper right')
#plt.xlim([44,50])
plt.xlabel(r"$L_{bol} (erg s^{-1})$")
#plt.title(r"$L_{{limit}}$ = {0:.2f}; completeness $z\in [0,3]$ = {1:.0f}%".format(L_limit, 100*(1-count1/len(z))))
plt.savefig("LumCut.png", dpi=300)'''
