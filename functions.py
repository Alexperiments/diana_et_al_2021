#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math
from scipy.optimize import curve_fit
from math import factorial
import scipy
from scipy import stats
import scipy.integrate as integrate

flag_calcola_errori = True #calcola gli errori con montecarlo, ci mette un sacco e sovrascrive il file uncertainties
flag_StoN = False  # produce il file StoNratio

calcola_di_nuovo_tutti_dati = False
Flag_spike = False #stampa immagini che rappresentano gli spike rimossi negli spettri con continuo rimosso e lo zscore
Flag_eps = True #stampa eps con la rimozione del continuo
Flag_masks = False # stampa gli spettri con assorbimenti e spike residui mascherati
Flag_baldwin = True # produce dei grafici per la correlazione EW - luminosità di 1350 e del radio

names = ['GB6J001115+144608','GB6J003126+150729','GB6J012126+034646','GB6J083548+182519','GB6J083945+511206','GB6J091825+063722','GB6J102107+220904','GB6J102623+254255','GB6J132512+112338','GB6J134811+193520','GB6J141212+062408','GB6J143023+420450','GB6J151002+570256','GB6J153533+025419','GB6J161216+470311','GB6J162956+095959','GB6J164856+460341','GB6J171103+383016','GB6J235758+140205']
z =  [4.94,4.29,4.13,4.41,4.4,4.17,4.26,5.27,4.40,4.38,4.37,4.72,4.31,4.37,4.33,4.97,5.36,3.98,4.25]
bg_int = [[1610,1640,1440,1450],
	[1620,1625,1435,1455],
	[1680,1710,1435,1465], #GB6J012126+034646
	[1600,1620,1345,1365], #GB6J083548+182519
	[1675,1695,1350,1365],
	[1680,1705,1352,1365],
	[1675,1700,1435,1465], #GB6J102107+220904
	[1325,1345,1435,1465], #GB6J102623+254255
	[1690,1710,1435,1465],
	[1690,1710,1435,1465],
	[1660,1680,1435,1465],
	[1690,1710,1435,1465],
	[1690,1710,1435,1465],
	[1690,1710,1435,1465],
    [1690,1710,1435,1465],
    [1629,1638,1435,1465],
	[1325,1345,1435,1455], #GB6J164856+460341
	[1640,1660,1350,1363],
	[1690,1710,1435,1445]]

masks_int = [[],
[],
[1455, 1493],
[],
[1516,1525],
[1516,1521,1529,1547, 1552, 1555],
[1695,1705],
[1506,1507.5],
[],
[],
[],
[1513,1518,1576,1588],
[],
[],
[],
[1557,1567],
[1535,1539,1558,1562],
[1517,1532],
[1445,1458,1554,1563]]

threshold = [2.6,2.6,2.6,2.6,2.6,2.6,2.6,2.6,2.6,2.6,2.6,2.6,2.6,2.6,2.6,2,2.6,2.6,2.5]

bound_dict = {'GB6J001115+144608': [[0,1535,0,0,1530,0],[50,1580,30,50,1580,30]],
'GB6J003126+150729': [[0,1500,0,0,1500,0],[100,1550,30,100,1550,30]],
'GB6J012126+034646': [[0,1500,1,0,1500,3,0,1500,3],[50,1560,50,50,1560,50,50,1560,50]],
'GB6J083548+182519': [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]],
'GB6J083945+511206': [[0,1525,0,0,1525,0],[50,1580,50,50,1580,50]],
'GB6J091825+063722': [[0,1525,0,0,1525,0],[50,1560,30,50,1560,30]],
'GB6J102107+220904': [[0,1500,0,0,1500,0],[50,1560,50,50,1555,50]],
'GB6J102623+254255': [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]],
'GB6J132512+112338': [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]],
'GB6J134811+193520': [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]],
'GB6J141212+062408': [[0,1520,0,0,1520,0],[50,1560,50,50,1560,25]],
'GB6J143023+420450': [[0,1540,1],[40,1560,30]],
'GB6J151002+570256': [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]],
'GB6J153533+025419': [[0,1500,10,0,1500,10],[50,1550,20,10,1570,15]],
'GB6J161216+470311': [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]],
'GB6J162956+095959': [[0,1535,0,0,1535,0],[50,1580,50,50,1580,50]],
'GB6J164856+460341': [[0,1535,0,0,1535,0],[50,1580,15,50,1580,15]],
'GB6J171103+383016': [[0,1500,0,0,1500,0],[50,1560,50,10,1560,30]],
'GB6J235758+140205': [[0,1520,10,0,1520,10],[60,1565,60,60,1565,60]]}

def gaus_func(x, a, x1, sigma1, b=0, x2=0, sigma2=1, c=0, x3=0, sigma3=1):
    return a*np.exp(-(x-x1)**2/(2*sigma1**2))+b*np.exp(-(x-x2)**2/(2*sigma2**2))+c*np.exp(-(x-x3)**2/(2*sigma3**2))

########################  SMOOTHING FUNCTION  ############################

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

######################  (END)SMOOTHING FUNCTION  ##########################

########################  CONTINUUM ESTIMATE  ############################

def continuum_estimate(data, background_int):
    int1 = data[1][np.where(data[0] > background_int[2])[0][0]:np.where(data[0] < background_int[3])[0][-1]]
    int2 = data[1][np.where(data[0] > background_int[0])[0][0]:np.where(data[0] < background_int[1])[0][-1]]
    y1 = np.mean(int1)
    y2 = np.mean(int2)
    x1 = (background_int[1]+background_int[0])*0.5
    x2 = (background_int[3]+background_int[2])*0.5
    m = (y2-y1)/(x1-x2)
    q = ((y1+y2) - m*(x1+x2))*0.5
    return m,q

######################  (END)CONTINUUM ESTIMATE  ##########################

########################  MASK SPECTRUM FUNCTION  ############################

def mask(x, y, masks):
    data = y.copy()
    if not masks:
        return y
    for j in range(int(len(masks)/2)):
        low = masks[2*j]
        high = masks[2*j+1]
        ymin = data[np.where(x > low)[0][0]]
        ymax = data[np.where(x < high)[0][-1]]
        m, q = continuum_estimate(np.vstack((x, data)), [high, high+5, low-5,low])
        cont = x*m + q
        noise1 = data[np.where(x < low)[0][-10]:np.where(x < low)[0][-1]]
        noise2 = data[np.where(x > high)[0][0]:np.where(x > high)[0][9]]
        mean_noise = (statistics.stdev(noise1) + statistics.stdev(noise2))*0.5
        for i in range(len(x)):
            if (i > np.where(x > low)[0][0]) and (i < np.where(x < high)[0][-1]):
                data[i] = cont[i] + np.random.normal(0., mean_noise, 1)
    return data

######################  (END)MASK SPECTRUM FUNCTION  ##########################

########################  DISTRUBUTION HISTOGRAM  ############################

def draw_hist(truevalue, data, flag=True):

	sLim=0.68 #Intervallo di confidenza
	nBins=25

	hist, bin_edges = np.histogram(data, bins=nBins)
	'''hist_dist = scipy.stats.rv_histogram(hist)
	NormLo = hist_dist.ppf(0.5-(sLim/2))
	NormHi = hist_dist.ppf(0.5+(sLim/2))
	Med = np.median(data)'''

	#Med = (bin_edges[hist.argmax()] + bin_edges[hist.argmax() + 1])*0.5
	Med = truevalue
	gHi = np.where(data >= Med)[0]
	gLo = np.where(data < Med)[0]
	vSortLo = np.sort(data[gLo])
	vSortHi = np.sort(data[gHi])
	NormLo = vSortLo[np.int((1.0-sLim)*np.size(vSortLo))]
	NormHi = vSortHi[np.int(sLim      *np.size(vSortHi))]
	if flag:
		plt.hist(data,bins=nBins,alpha=0.5, color='g', ec='black', linewidth='0.25')
		plt.xlabel(r'Mass ($M_{\odot}$)')

		plt.axvline(x=Med, c='red', ls='--', lw=1)
		plt.axvline(x=NormLo, c='black', ls='-.', lw=1)
		plt.axvline(x=NormHi, c='black', ls='-.', lw=1)

		plt.annotate('%i percent limits' % (sLim*100), (0.6,0.9), xycoords='axes fraction')
		plt.title('Mass: <%.2f> -%.2f +%.2f' % (Med, Med-NormLo, NormHi-Med))

	return Med-NormLo, NormHi-Med

######################  (END)DISTRUBUTION HISTOGRAM  ##########################

########################  ASYMETTRICAL DISTRUBUTION  ############################

def random_asy_distr(media, errl, errh, Ntries):
	norm = -errh/errl
	dlow = np.zeros(int(Ntries/(norm+1))+1)
	dhigh = np.zeros(int(Ntries*norm/(norm+1)))
	i = 0
	while dlow[-1] == 0:
		a = np.random.normal(media, -errl)
		if a <= media:
			dlow[i] = a
			i += 1
	i = 0
	while dhigh[-1] == 0:
		a = np.random.normal(media, errh)
		if a >= media:
			dhigh[i] = a
			i += 1
	return np.concatenate((dlow, dhigh))[0:Ntries]

######################  (END)ASYMETTRICAL DISTRUBUTION  ##########################

def fit_CIV(name, x, y):
    init_guess = [3, 1549, 10, 2, 1549, 15]
    x_unity = np.arange(1350,1750,0.1)
    if name == 'GB6J143023+420450':
        init_guess = [3, 1549, 10]
        par, cov = curve_fit(gaus_func, x, y, p0=init_guess, absolute_sigma=True, bounds=bound_dict[name])
        ym = gaus_func(x, par[0], par[1], par[2])
        ycalc = gaus_func(x_unity, par[0], par[1], par[2])
        integral = math.sqrt(2*math.pi)*(par[0]*par[2])
        centroid = (par[0]*par[1]*par[2])/(par[0]*par[2])
        second_moment = math.sqrt(2*math.pi)*((par[1]*par[1]+par[2]*par[2])*par[0]*par[2])/integral - centroid*centroid
        delta_v = (1549 - par[1])*299792/1549
        median = x[find_median_index(ym)]
    elif name == 'GB6J012126+034646':
        init_guess = [3, 1549, 10,0,1540,5,40,1560,30]
        par, cov = curve_fit(gaus_func, x, y, p0=init_guess, absolute_sigma=True, bounds=bound_dict[name])
        ym = gaus_func(x, par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8])
        ycalc = gaus_func(x_unity, par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8])
        integral = math.sqrt(2*math.pi)*(par[0]*par[2]+par[3]*par[5]+par[6]*par[8])
        centroid = (par[0]*par[1]*par[2]+par[3]*par[4]*par[5]+par[6]*par[7]*par[8])/(par[0]*par[2]+par[3]*par[5]+par[6]*par[8])
        second_moment = math.sqrt(2*math.pi)*((par[1]*par[1]+par[2]*par[2])*par[0]*par[2] + (par[4]*par[4]+par[5]*par[5])*par[3]*par[5] + (par[7]*par[7]+par[8]*par[8])*par[6]*par[8])/integral - centroid*centroid
        min_lambda = min(par[1], par[4], par[7])
        delta_v = (1549 - min_lambda)*299792/1549
        median = x[find_median_index(ym)]
    else:
        par, cov = curve_fit(gaus_func, x, y, p0=init_guess, absolute_sigma=True, bounds=bound_dict[name])
        ym = gaus_func(x, par[0], par[1], par[2], par[3], par[4], par[5])
        ycalc = gaus_func(x_unity, par[0], par[1], par[2], par[3], par[4], par[5])
        integral = math.sqrt(2*math.pi)*(par[0]*par[2]+par[3]*par[5])
        centroid = (par[0]*par[1]*par[2]+par[3]*par[4]*par[5])/(par[0]*par[2]+par[3]*par[5])
        second_moment = math.sqrt(2*math.pi)*((par[1]*par[1]+par[2]*par[2])*par[0]*par[2] + (par[4]*par[4]+par[5]*par[5])*par[3]*par[5])/integral - centroid*centroid
        min_lambda = min(par[1], par[4])
        delta_v = (1549 - min_lambda)*299792/1549
        median = x[find_median_index(ym)]
    return ym, ycalc, integral, median, np.sqrt(second_moment)*299792/1549, delta_v

def find_median_index(y):
	cumulative = np.cumsum(y)
	half = max(cumulative)/2
	new = np.where(cumulative > half, cumulative, 0)
	for i,elem in enumerate(new):
		if elem > 0:
			return i

########################  COSMOLOGICAL CALCULATOR  ############################

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

######################  (END)COSMOLOGICAL CALCULATOR  ##########################

def fit_A():
    global fit
    fit = 'A'
    #Global fit A
    global a0, a1, a2
    global b0, b1, b2
    global c0, c1, c2
    global d0, d1
    global x0
    a0, a1, a2 = [0.8569, -0.2614, 0.0200]
    b0, b1, b2 = [2.5375, -1.0425, 1.1201]
    c0, c1, c2 = [13.0088, -0.5759, 0.4554]
    d0, d1 = [-3.5426, -0.3936]
    x0 = 2
def fit_B():
    global fit
    fit = 'B'
    #Global fit B
    global a0, a1, a2
    global b0, b1, b2
    global c0, c1, c2
    global d0, d1
    global x0
    a0, a1, a2 = [0.3653, -0.6006, 0]
    b0, b1, b2 = [2.4709, -0.9963, 1.0716]
    c0, c1, c2 = [12.9656, -0.5758, 0.4698]
    d0, d1 = [-3.6276, -0.3444]
    x0 = 2

fit_A()

#Chebyshev polynomials 0,1,2-th orders
def T_0(x):
    return 1
def T_1(x):
    return x
def T_2(x):
    return 2*x*x-1

def gamma_1(x):
    return a0*T_0(1+x) + a1*T_1(1+x) + a2*T_2(1+x)
def gamma_1_Hopkins(x):
    return a0*((1+x)/(1+x0))**a1
def gamma_2(x):
    return 2*b0/( ((1+x)/(1+x0))**b1 + ((1+x)/(1+x0))**b2 )
def logL_star(x):
    return 2*c0/( ((1+x)/(1+x0))**c1 + ((1+x)/(1+x0))**c2 )
def logPhi_star(x):
    return d0*T_0(x) + d1*T_1(x)
def L_star(x):
    return 10**(2*c0/( ((1+x)/(1+x0))**c1 + ((1+x)/(1+x0))**c2 ))
def Phi_star(x):
    return 10**(d0*T_0(x) + d1*T_1(x))

def log_Phi_bol(L,z):
    if fit == 'A':
        return logPhi_star(1 + z) - gamma_1(z)*(L - logL_star(z)) - gamma_2(z)*(L-logL_star(z))
    if fit == 'B':
        return logPhi_star(1 + z) - gamma_1_Hopkins(z)*(L - logL_star(z)) - gamma_2(z)*(L-logL_star(z))
def Phi_bol(L,z):
    if fit == 'A':
        return Phi_star(1 + z)/(((L/L_star(z))**gamma_1(z) + (L/L_star(z))**gamma_2(z)))
    elif fit == 'B':
        return Phi_star(1 + z)/(((L/L_star(z))**gamma_1_Hopkins(z) + (L/L_star(z))**gamma_2(z)))

def trap_int_L(Lminim, Lmaxim, z):
    Lmin = 10**(Lminim -np.log10(3.9) - 33)
    Lmax = 10**(Lmaxim -np.log10(3.9) - 33)
    L_bin = np.linspace(Lmin, Lmax, int(100000*(Lmaxim-Lminim)))
    result = np.zeros(np.size(z))
    dL = L_bin[1] - L_bin[0]
    for i,z_bin in enumerate(z):
        result[i] = integrate.trapz(Phi_bol(L_bin,z_bin)/L_bin, dx=dL)
    return np.log10(result) + 9

def M_in_Lbol(M, Redd):
    return M + np.log10(1.26) + 38 + np.log10(Redd)
