#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics

from astropy.modeling import models
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.integrate import quad

##########################	COSTANTI 	    ##########################

H0 = 70 #km/s/Mpc
Omega = 0.3 #Frazione di materia Lambda-CDM
c = 299792.458 #km/s
pi = math.pi
G = 6.67e-8 #cme3 ge-1 se-2
mp = 1.67e-24 #g
sigma_T = 6.65e-25 # cme-2

def gaus_func(x, a, x1, sigma1, b=0, x2=0, sigma2=1, c=0, x3=0, sigma3=1):
	return a*np.exp(-(x-x1)**2/(2*sigma1**2))+b*np.exp(-(x-x2)**2/(2*sigma2**2))+c*np.exp(-(x-x3)**2/(2*sigma3**2))

def luminosity(flux, z):		     
	dl = ned_calc(z)
	inv_lum1350 = flux*1350*4*math.pi*dl**2

	return math.log10(inv_lum1350)

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
		adot = math.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
		age = age + 1./adot

	zage = az*age/n
	zage_Gyr = (Tyr/H0)*zage
	DTT = 0.0
	DCMR = 0.0

	# do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
	for i in range(n):
		a = az+(1-az)*(i+0.5)/n
		adot = math.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
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
	x = math.sqrt(abs(WK))*DCMR
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
	DL_Gyr = (Tyr/H0)*DL

	# comoving volume computation

	return DL_Mpc

######################  (END)COSMOLOGICAL CALCULATOR  ##########################

names = ['GB6J001115p144608','GB6J003126p150729','GB6J012126p034646','GB6J083548p182519','GB6J083945p511206','GB6J091825p063722','GB6J102107p220904','GB6J102623p254255','GB6J132512p112338','GB6J134811p193520','GB6J141212p062408','GB6J143023p420450','GB6J151002p570256','GB6J153533p025419','GB6J161216p470311','GB6J162956p095959','GB6J164856p460341','GB6J171103p383016','GB6J235758p140205']
z =  [4.94,4.29,4.13,4.41,4.4,4.17,4.26,5.27,4.40,4.38,4.37,4.72,4.31,4.37,4.33,4.97,5.36,3.98,4.25]
bounds = [[[0,1500,0,0,1500,0],[40,1580,40,5,1580,40]], [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]], [[0,1500,1,0,1500,3,0,1500,3],[50,1580,50,50,1580,50,50,1580,50]], [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]], [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]], [[0,1535,0,0,1535,0],[50,1580,25,50,1580,25]], [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]], [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]], [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]], [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]], [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]], [[0,1540,1],[40,1560,30]], [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]], [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]], [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]], [[0,1535,0,0,1535,0],[50,1580,50,50,1580,50]], [[0,1535,0,0,1535,0],[50,1580,15,50,1580,15]], [[0,1500,0,0,1500,0],[50,1580,50,50,1580,50]], [[0,1530,0,0,1530,0],[50,1580,50,50,1580,50]]]

Ntries = 1000

dati_file = 'daticompl.txt'
output = open('uncertaintiess.txt', 'w+')
output.write('Name\tredshift\tvelocity\tvel unc\tlum1350\tLum1350 unc\n')

daticompl = open(dati_file, 'r')
qdp_velocities = []
qdp_errors = []
qdp_fwhm = []
qdp_fwhm_uncert = []
_ = daticompl.readline()
_ = daticompl.readline()
for i in range(19):
	line = daticompl.readline()
	temp = line.split()
	qdp_velocities.append(float(temp[1]))
	qdp_errors.append(float(temp[2]))
	qdp_fwhm.append(float(temp[3]))
	qdp_fwhm_uncert.append(float(temp[4]))
py_velocities = []
py_errors = []
py_fwhm = []
py_fwhm_uncert = []

for j in range(19):
	print('processing '+names[j]+'...')
	qdp_file = names[j]+'_.qdp'
	
	output.write(names[j])

	fp = open(qdp_file, 'r')
	lines = fp.readlines()[2:]

	x = np.zeros(len(lines))
	y = np.zeros(len(lines))
	pow1455 = np.zeros(len(lines))
	
	for i, line in enumerate(lines):	
		x[i] = float(line.split()[0])
		y[i] = float(line.split()[1])

	fp.close()
	
	init_guess = [3, 1549, 10, 2, 1549, 15]
	x_unity = np.arange(1350,1750,1)

	if names[j] == 'GB6J143023p420450':
		init_guess = [3, 1549, 10]
		par, cov = curve_fit(gaus_func, x, y, p0=init_guess, absolute_sigma=True, bounds=bounds[j])
		ym = gaus_func(x, par[0], par[1], par[2])
		ycalc = gaus_func(x_unity, par[0], par[1], par[2])
		integral = math.sqrt(2*math.pi)*(par[0]*par[2])
	elif names[j] == 'GB6J012126p034646':
		init_guess = [3, 1549, 10,0,1540,5,40,1560,30]
		par, cov = curve_fit(gaus_func, x, y, p0=init_guess, absolute_sigma=True, bounds=bounds[j])
		ym = gaus_func(x, par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8])
		ycalc = gaus_func(x_unity, par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8])
		integral = math.sqrt(2*math.pi)*(par[0]*par[2]+par[3]*par[5]+par[6]*par[8])
	else:
		par, cov = curve_fit(gaus_func, x, y, p0=init_guess, absolute_sigma=True, bounds=bounds[j])
		ym = gaus_func(x, par[0], par[1], par[2], par[3], par[4], par[5])
		ycalc = gaus_func(x_unity, par[0], par[1], par[2], par[3], par[4], par[5])
		integral = math.sqrt(2*math.pi)*(par[0]*par[2]+par[3]*par[5])
			
	num = sum(np.multiply(x_unity**3, ycalc))
	num2 = sum(np.multiply(x_unity**2, ycalc))
	den = sum(np.multiply(x_unity, ycalc))
	gmax = max(ycalc)
	lamb = [l for i, l in enumerate(x_unity) if ycalc[i] >= gmax/2]
	best_second_moment = math.sqrt(num/den - (num2/den)**2)
	best_line_luminosity = luminosity(integral*1e-17, z[j])
	best_fwhm = (max(lamb)-min(lamb))*299792/1549

	x = np.asarray(x)
	continuum = y - ym
	noise = statistics.stdev(continuum[0:150])

	second_moment = np.zeros(Ntries)
	integrals = np.zeros(Ntries)
	fwhms = np.zeros(Ntries)

	print('mocking '+names[j]+'...')
	for i in range(Ntries):	
		ymock = ym + np.random.normal(0., noise, x.shape)
		if names[j] == 'GB6J143023p420450':
			init_guess = [3, 1549, 10]
			par, cov = curve_fit(gaus_func, x, ymock, p0=init_guess, absolute_sigma=True, bounds=bounds[j])
			mock_model = gaus_func(x_unity, par[0], par[1], par[2])
			integral = math.sqrt(2*math.pi)*(par[0]*par[2])
		elif names[j] == 'GB6J012126p034646':
			init_guess = [3, 1549, 10,0,1540,5,40,1560,30]
			par, cov = curve_fit(gaus_func, x, ymock, p0=init_guess, absolute_sigma=True, bounds=bounds[j])
			mock_model = gaus_func(x_unity, par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8])
			integral = math.sqrt(2*math.pi)*(par[0]*par[2]+par[3]*par[5]+par[6]*par[8])
		else:
			par, cov = curve_fit(gaus_func, x, ymock, p0=init_guess, absolute_sigma=True, bounds=bounds[j])
			mock_model = gaus_func(x_unity, par[0], par[1], par[2], par[3], par[4], par[5])
			integral = math.sqrt(2*math.pi)*(par[0]*par[2]+par[3]*par[5])

		num = sum(np.multiply(x_unity**3, mock_model))
		num2 = sum(np.multiply(x_unity**2, mock_model))
		den = sum(np.multiply(x_unity, mock_model))
		gmax = max(mock_model)
		lamb = [l for i, l in enumerate(x_unity) if mock_model[i] >= gmax/2]
		
		second_moment[i] = math.sqrt(num/den - (num2/den)**2)
		fwhms[i] = (max(lamb)-min(lamb))
		integrals[i] = integral
	
	_ = plt.hist(integrals, bins='auto')  # arguments are passed to np.histogram
	plt.title(names[j])
	plt.savefig(names[j]+'_hist.png')
	plt.clf()

	velocity = 299792*best_second_moment/1549
	velocity_uncertainty = 299792*statistics.stdev(second_moment)/1549
	luminosity_uncertainty = luminosity(statistics.stdev(integrals*1e-17), z[j])
	luminosity_uncertainty = 10**(luminosity_uncertainty-best_line_luminosity)
	fwhm_uncertainty = 299792*statistics.stdev(fwhms)/1549

	output.write('\t%.2f' % z[j])
	output.write('\t%.0f' % velocity)
	output.write('\t%.0f' % velocity_uncertainty)
	output.write('\t%.0f' % best_fwhm)
	output.write('\t%.0f' % fwhm_uncertainty)
	output.write('\t%.2f' % best_line_luminosity)
	output.write('\t%.2f\n' % luminosity_uncertainty)
	py_velocities.append(velocity)
	py_errors.append(velocity_uncertainty)
	py_fwhm.append(best_fwhm)
	py_fwhm_uncert.append(fwhm_uncertainty)
output.close()	

x = np.arange(10000)
y = x
plt.errorbar(qdp_fwhm,py_fwhm,xerr=qdp_fwhm_uncert, yerr=py_fwhm_uncert, fmt='none')
plt.plot(x,y)
plt.tick_params(axis ='x', rotation = 20) 
plt.xlabel('velocity (km/s)')
plt.ylabel('velocity (km/s)')
plt.show()