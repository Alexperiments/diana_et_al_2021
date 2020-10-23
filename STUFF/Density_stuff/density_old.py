#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import statistics 
import math

from math import *
from astropy.modeling import models
from astropy import units as u
from scipy.optimize import curve_fit

##########################	COSTANTI 	    ##########################

H0 = 70 #km/s/Mpc
Omega_m = 0.3 #Frazione di materia Lambda-CDM
Omega_vac = 0.7 #Frazione di dark energy Lambda-CDM
c = 299792.458 #km/s
pi = math.pi
G = 6.67e-8 #cme3 ge-1 se-2
mp = 1.67e-24 #g
sigma_T = 6.65e-25 # cme-2
alpha = 0.44 # k-correction

##########################	  FLAGS 	  ############################

corr_shen = True
corr_ID = True

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

	return DL_Mpc, DCMR_Mpc, V_Gpc

######################  (END)COSMOLOGICAL CALCULATOR  ##########################

def write_counts(x_redshift, count, sky_cover_area):
	'''Crea un file 'conteggio_per_sm.txt' con i seguenti dati:
			z medio bin   #   Oggetti con massa > 10^9   #   Larghezza del bin (una riga)
	'''
	out_sm = open('conteggio_per_sm.txt','w+')
	out_sm.write("{0}\t{1:.2f}\t{2}\n".format(x_redshift[0], count[0], sky_cover_area))
	for i in range(1,len(count)):
		out_sm.write("{0}\t{1:.2f}\n".format(x_redshift[i], count[i]))
	out_sm.write(str(x_redshift[-1]))
	out_sm.close()
	
def sdss_compl_calc(z_min, z_max):
	''' Calcola la frazione di oggetti non usati da Shen ma comunque presenti nella SDSS in un determinato bin
	'''
	sdss_completeness_corr = 1
	if corr_shen:
		found = 0
		lines = 0
		with open('sdss_in_class.txt') as sdss:
			next(sdss)
			mean_z = (z_min+z_max)*0.5
			flux_limit, mag_limit = flux_mag_limits(mean_z) 
			for line in sdss:
				string = line.split()				
				name = string[0]								
				z_SDSS = float(string[5])
				if ((z_SDSS >= z_min) & (z_SDSS <= z_max) & (float(string[3]) == 1) & (float(string[1]) >= flux_limit) & (float(string[2]) <= mag_limit)):
					lines += 1
					if float(string[6]) > 0:
						found += 1 
		sdss_completeness_corr = lines/found 
		print("Totale = {} / Con massa = {} -> Shen_corr = {}".format(lines,found,sdss_completeness_corr))
	return sdss_completeness_corr
	
def ID_compl_calc(flux_limit, mag_limit):
#Corregge per gli oggetti con redshift nella SDSS 
	ID_compl_corr = 1
	if corr_ID:
		found = 0
		lines = 0
		with open('sdss_in_class.txt') as sdss:
			next(sdss)
			for line in sdss:
				string = line.split()
				name = string[0]								
				if (float(string[1]) >= flux_limit) & (float(string[2]) <= mag_limit) & (float(string[2]) > 0) & is_in_area(name):
					lines += 1
					if float(string[4]) > 0:
						found += 1 
	ID_compl_corr = lines/found 
	print("Totale = {} / Con ID = {} -> ID_corr = {}".format(lines,found,ID_compl_corr))
	return ID_compl_corr

def flux_mag_limits(z):
	dist_z , _, _ = ned_calc(z, H0, Omega_m, Omega_vac)
	dist_ratio = dist4_5/dist_z
	flux_limit = 30*dist_ratio**2*((1+4.5)/(1+z))**(k_corr)																		
	mag_limit = 22.25-5*math.log10(dist_ratio)-2.5*math.log10(((1+4.5)/(1+z))**(k_corr))
	return flux_limit,mag_limit,
	
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

def AR_decl_in_deg(stringa):
	stringa = stringa.replace('GB6J', '')
	stringa = stringa.split('+')

	if (len(stringa) == 2):
		AR=stringa[0]
		stringa2 = stringa[1].split()
		decl=stringa2[0]
	else:
		stringa = line.split('-')
		AR=stringa[0]
		stringa2 = stringa[1].split()
		decl=stringa2[0]
	stringa = line.split()

	digits = [i for i in AR]
	ora = float(digits[0]+digits[1])
	minuti = float(digits[2]+digits[3])
	secondi = float(digits[4]+digits[5])
	decimal_AR = (ora + minuti/60 + secondi/3600)
	decimal_AR = decimal_AR*360/24
	
	digits = [i for i in decl]
	gradi = float(digits[0]+digits[1])
	minuti = float(digits[2]+digits[3])
	secondi = float(digits[4]+digits[5])
	decimal_decl = (gradi + minuti/60 + secondi/3600)
	
	return decimal_AR,decimal_decl

name = []
z = []
z_massless = []
radio_flux = []
flag_class = []
mag = []
m_shen = []
AR = []
decl = []

with open('sdss_in_class.txt') as fp:
	next(fp)
	for line in fp:
		string = line.split()
		if (float(string[6]) > 0):
			name.append(string[0])
			z.append(float(string[4]))
			flag_class.append(float(string[3]))
			radio_flux.append(float(string[1]))
			mag.append(float(string[2]))
			m_shen.append(float(string[6]))	

#La selezione dell'area viene tenuta usata solo per la correzione_ID, considerando la più grande area in cui sia la CLASS che la SDSS sono presenti
decl_min = '000000'
decl_max = '600000'
AR_min = '080000'
AR_max = '160000'
sky_cover_area = 11500   #sq. degree
bin_width = 0.5
z_min = 0.75
z_max = 3.75
k_corr = alpha-1

z = np.asarray(z)
m_shen = np.asarray(m_shen)
z_bin = np.arange(z_min, z_max+bin_width, bin_width)
count = np.zeros(len(z_bin)-1)
density_err = np.zeros([2,len(z_bin)-1])
density = np.zeros(len(z_bin)-1)
frac_area = sky_cover_area*3.0462e-4/(4*pi)
norm_comov_vol = np.empty([len(z), 1])

dist4_5, _, _ = ned_calc(4.5, H0, Omega_m, Omega_vac)

limit_radio = math.log10(30*4*math.pi) - 26 + 2*math.log10(3.06*dist4_5) + 48 + math.log10((1+4.5)**(-0.56)) + 9 + math.log10(5)
flusso_ottico = 5.4*math.sinh(22.8435 - 0.921034*21.5) # Ricavata dalla relazione della asinh magnitude della SDSS, flusso di punto zero (banda r) = 2.25e-9
limit_1400 = math.log10(flusso_ottico) - 19 + math.log10(4*math.pi) + 2*math.log10(3.09*dist4_5) + 48 + (k_corr)*math.log10(1+4.5) + 15.33

for i in range(len(z_bin)-1):
	condition = [(z >= z_bin[i]) & (z < z_bin[i+1])]	#seleziona i redshift compresi nell' i-esimo bin
	choselist = [z]
	selected_z = np.select(condition, choselist)
	print("z {} - {}".format(z_bin[i],z_bin[i+1]))
	mean_z = (z_bin[i+1] + z_bin[i])*0.5														#redshift medio del bin
	_, _, vol_min = ned_calc(z_bin[i], H0, Omega_m, Omega_vac)			# volume inferiore della shell
	_, _, vol_max = ned_calc(z_bin[i+1], H0, Omega_m, Omega_vac)		# volume superiore della shell

	norm_comov_vol[i] = (vol_max-vol_min)*frac_area									# volume della shell normalizzato per l'area coperta dalla survey
			
	for j in range(len(selected_z)):																# applica i tagli in flusso radio e magnitudine 
		if (selected_z[j] > 0) & (m_shen[j] >= 1) & (flag_class[j] == 1):
			distMpc, _, _ = ned_calc(z[j], H0, Omega_m, Omega_vac)
			L_radio = math.log10(radio_flux[j]*4*math.pi) - 26 + 2*math.log10(3.09*distMpc) + 48 + math.log10((1+z[j])**(k_corr)) + 9 + math.log10(5)
			flusso_ottico = 5.4*math.sinh(22.8435 - 0.921034*mag[j])
			L_1400 = math.log10(flusso_ottico) - 19 + math.log10(4*math.pi) + 2*math.log10(3.09*distMpc) + 48 + math.log10((1+z[j])**(k_corr)) + 15.33 + (k_corr)*math.log10(1400/6580)
			
			limit_radio = 43
			
			if  (L_radio >= limit_radio): # & (L_1400 >= limit_1400):
				count[i] += 1	
	print("Conteggi reali = {}".format(count[i]))
	sdss_completeness_corr = sdss_compl_calc(z_bin[i], z_bin[i+1])     # coefficiente di completezza SDSS/Shen (se Shen ha calcolato la massa di tutti gli oggetti presenti nella SDSS in un certo bin, coeff == 1)							
	#count[i] *= sdss_completeness_corr     
	flux_limit, mag_limit = flux_mag_limits(mean_z)                        
	ID_compl_corr = ID_compl_calc(flux_limit, mag_limit)							 # moltiplica per un coef correttivo dovuto alla mancanza nel catalogo SDSS di alcuni oggetti CLASS (~40%)
	#count[i] *= ID_compl_corr
	
	print("Conteggi corretti = {}\n".format(count[i]))
	
	count_uerr = count[i] + math.sqrt(count[i])
	count_ierr = count[i] - math.sqrt(count[i])
	if count[i] != 0:
		density[i] = math.log10(count[i]/norm_comov_vol[i])
		density_err[0][i] = math.log10(count_uerr/norm_comov_vol[i]) - math.log10(count[i]/norm_comov_vol[i])
		density_err[1][i] = math.log10(count[i]/norm_comov_vol[i]) - math.log10(count_ierr/norm_comov_vol[i])
	else :
		density[i] = -1.75
		
write_counts(z_bin, count, sky_cover_area)

fig, ax = plt.subplots(figsize=(9, 5))	
for i in z_bin:
	plt.axvline(x=i, linestyle=':', color='grey')	
x_redshift = np.arange(z_min+bin_width*0.5, z_max+bin_width*0.5, bin_width)
plt.title(r'Blazar space density with $L_{{5GHz}}\nu > 10^{{43}}erg s^{{-1}}$ Sky coverage = {} $deg^2$'.format(sky_cover_area)) 
ax.errorbar(x_redshift, density, yerr=density_err, fmt='o', label='CLASS+Shen')
#ax.scatter([4.5, 5.5], [math.log10(0.074), math.log10(0.017)], label='CLASS', c='r') #Punti CLASS
#ax.scatter(3.5, -0.801, label='BAT', marker='x', c='g')																				#Punti BAT
plt.xlabel('redshift')	
plt.ylabel('$Log\Phi(Gpc^{-3})$')
plt.legend()
fig.savefig("space_density_{}.png".format(sky_cover_area))