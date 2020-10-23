#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math

from math import *
from astropy.modeling import models
from astropy import units as u
from scipy.optimize import curve_fit

###########################  	COSTANTI   #############################

H0 = 70 #km/s/Mpc
Omega_m = 0.3 #Frazione di materia Lambda-CDM
Omega_vac = 0.7 #Frazione di dark energy Lambda-CDM
c = 299792.458 #km/s
pi = math.pi
G = 6.67e-8 #cme3 ge-1 se-2
mp = 1.67e-24 #g
sigma_T = 6.65e-25 # cme-2

###########################  	PARAMETRI   #############################

#La selezione dell'area viene tenuta usata solo per la correzione_ID, considerando la più grande area in cui sia la CLASS che la SDSS sono presenti
decl_min = '000000'
decl_max = '600000'
AR_min = '080000'
AR_max = '160000'
sky_cover_area = 11000
bin_width = 1
z_min = 0
z_max = 5
alpha = 0.44 # spectral index
k_corr = alpha-1
bol_corr = 3.81 # correzione per L_bol -> L_1400*1400
Edd_ratio = 0.2 # Edd_ratio impostato come limite di completezza
L_bol = 1.2*10.0**(47.0) #L_bol di un SMBH 10^9M a edd_ratio = 1
mass_limit = 9

############################  FUNZIONI UTILI  ################################

def write_counts(x_redshift, count, sky_cover_area):
	'''Crea un file 'conteggio_per_sm.txt' con i seguenti dati:
			z medio bin   #   Oggetti con massa > 10^9   #   Larghezza del bin (una riga)
	'''
	out_sm = open('conteggi.txt','w+')
	out_sm.write("{0}\t{1:.2f}\t{2}\n".format(x_redshift[0], count[0], sky_cover_area))
	for i in range(1,len(count)):
		out_sm.write("{0}\t{1:.2f}\n".format(x_redshift[i], count[i]))
	out_sm.write(str(x_redshift[-1]))
	out_sm.close()

def is_in_area(name):
	stringa = name.split('+')
	if (len(stringa) == 2):
		ARt = stringa[0]
		stringa2 = stringa[1].split()
		declt = stringa2[0]
	else:
		stringa = name.split('-')
		ARt = stringa[0]
		stringa2 = stringa[1].split()
		declt = stringa2[0]
	ARt = ARt.replace('GB6J', '')
	return ((ARt <= AR_max) & (ARt >= AR_min) & (declt >= decl_min) & (declt <= decl_max))

########################  COSMOLOGICAL CALCULATOR  ############################

def ned_calc(z, H0, Omega_m, Omega_vac):
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
		adot = sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
		age = age + 1./adot

	zage = az*age/n
	zage_Gyr = (Tyr/H0)*zage
	DTT = 0.0
	DCMR = 0.0

	# do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
	for i in range(n):
		a = az+(1-az)*(i+0.5)/n
		adot = sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
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
	x = sqrt(abs(WK))*DCMR
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

	ratio = 1.00
	x = sqrt(abs(WK))*DCMR
	if x > 0.1:
		if WK > 0:
		  ratio = (0.125*(exp(2.*x)-exp(-2.*x))-x/2.)/(x*x*x/3.)
		else:
		  ratio = (x/2. - sin(2.*x)/4.)/(x*x*x/3.)
	else:
		y = x*x
		if WK < 0: y = -y
		ratio = 1. + y/5. + (2./105.)*y*y
	VCM = ratio*DCMR*DCMR*DCMR/3.
	V_Gpc = 4.*pi*((0.001*c/H0)**3)*VCM

	return DL_Mpc

######################  (END)COSMOLOGICAL CALCULATOR  ##########################

def sdss_compl_calc(z_min, z_max):
	''' Calcola la frazione di oggetti non usati da Shen ma comunque presenti nella SDSS in un determinato bin
	'''
	found = 0
	total = 0
	with open('sdss_in_class1.txt') as sdss:
		next(sdss)
		mean_z = (z_min + z_max)*0.5
		flux_limit, mag_limit = flux_mag_limits(mean_z)
		for line in sdss:
			string = line.split()
			z_SDSS = float(string[5])
			if (z_SDSS >= z_min) & (z_SDSS < z_max) & (float(string[3]) == 1):
				if (float(string[2]) < mag_limit) & ( float(string[1]) > flux_limit):
					total += 1
					if float(string[6]) > 0:
						found += 1
	if found != 0: sdss_completeness_corr = total/found
	else: sdss_completeness_corr = 1
	print("Totale = {} / Con massa = {} -> Shen_corr = {}".format(total,found,sdss_completeness_corr))
	return sdss_completeness_corr

def ID_compl_calc(flux_limit, mag_limit):
	''' Calcola la frazione di oggetti presenti nella CLASS vs quelli con un identificativo (quindi un redshift)
	'''
	found = 0
	total = 0
	with open('sdss_in_class1.txt') as sdss:
		next(sdss)
		for line in sdss:
			string = line.split()
			if (float(string[1]) >= flux_limit) & (float(string[2]) <= mag_limit) & (float(string[2]) > 0) & is_in_area(string[0]):
				total += 1
				if float(string[4]) > 0:
					found += 1
	if found != 0: ID_compl_corr = total/found
	else: ID_compl_corr = 1
	print("Totale = {} / Con ID = {} -> ID_corr = {}".format(total,found,ID_compl_corr))
	return ID_compl_corr

def flux_mag_limits(z):
	dist_z = ned_calc(z, H0, Omega_m, Omega_vac)
	dist_ratio = dist4_5/dist_z
	# Calcolo dei limiti di luminosita (L_nu * nu) a cui siamo sensibili a z=4.5, i limiti sono applicati poi a tutti i bin
	ff = Edd_ratio*L_bol*(0.000000140/299792458.0)/(3.8)*(1+4.5)**(0.56)*(1400.0/6160.0)**(-0.44)/(4*pi*(dist4_5*3.9*10.0**(24.0))**2.0)
	magn4_5 = -2.5*math.log10(ff) - 48.6
	radio_limit = 30*dist_ratio**2*((1+4.5)/(1+z))**(k_corr)
	mag_limit = magn4_5-5*math.log10(dist_ratio)-2.5*k_corr*math.log10((1+4.5)/(1+z))
	return radio_limit,mag_limit,

num_sources = len(open('sdss_in_class1.txt').readlines(  ))

name 			 = []
z					 = np.zeros(num_sources-1)
z_SDSS		 = np.zeros(num_sources-1)
radio_flux = np.zeros(num_sources-1)
flag_class = np.zeros(num_sources-1)
mag 			 = np.zeros(num_sources-1)
m_shen 		 = np.zeros(num_sources-1)
edd_ratio  = np.zeros(num_sources-1)
############################ LETTURA FILE INPUT ################################

with open('sdss_in_class1.txt') as fp:
	''' Salva tutti gli oggetti con massa calcolata da Shen (e mag non nulla)
	'''
	next(fp)
	for i in range(num_sources-1):
		string = fp.readline().split()
		if (float(string[6]) > mass_limit) & (float(string[2]) > 0):
			name.append(string[0])
			radio_flux[i] = float(string[1])
			mag[i] 				= float(string[2])
			flag_class[i]	= float(string[3])
			z[i]					= float(string[4])
			z_SDSS[i] 		= float(string[5])
			m_shen[i] 		= float(string[6])
			edd_ratio[i]  = float(string[8])
		else: name.append('')

z_bin = np.arange(z_min, z_max+bin_width, bin_width) 	# array contenente z_min e z_max di ogni bin
count = np.zeros(len(z_bin)-1)
dist4_5 = ned_calc(4.5, H0, Omega_m, Omega_vac) # distanza di luminosità a z=4.5

################################# CALCOLO DEI CONTEGGI ##############################

for i in range(len(z_bin)-1):
	condition = [(z > z_bin[i]) & (z < z_bin[i+1])]	#seleziona i redshift compresi nell' i-esimo bin
	choselist = [z]
	selected_z = np.select(condition, choselist)

	print("z {} - {}".format(z_bin[i],z_bin[i+1]))
	mean_z = (z_bin[i+1] + z_bin[i])*0.5														#redshift medio del bin

	for j in range(len(selected_z)):																# applica i tagli in luminosità radio e ottica
		if (selected_z[j] > 0) & (flag_class[j] == 1):
			flux_limit, mag_limit = flux_mag_limits(selected_z[j])
			test = edd_ratio[j] + m_shen[j]
			#if (mag[j] < mag_limit) & (radio_flux[j] > flux_limit) :
			if (radio_flux[j] > flux_limit) & (test >= math.log10(Edd_ratio) + mass_limit):
				count[i] += 1
	print(mag_limit)
	print(flux_limit)
	print("Conteggi reali = {}".format(count[i]))

	sdss_completeness_corr = sdss_compl_calc(z_bin[i], z_bin[i+1])     # coefficiente di completezza SDSS/Shen (se Shen ha calcolato la massa di tutti gli oggetti presenti nella SDSS in un certo bin, coeff == 1)
	count[i] *= sdss_completeness_corr
	flux_limit, mag_limit = flux_mag_limits(mean_z)
	ID_compl_corr = ID_compl_calc(flux_limit, mag_limit)							 # moltiplica per un coef correttivo dovuto alla mancanza nel catalogo SDSS di alcuni oggetti CLASS (~40%)
	count[i] *= ID_compl_corr

	print("Conteggi corretti = {}\n".format(count[i]))

write_counts(z_bin, count, sky_cover_area)
