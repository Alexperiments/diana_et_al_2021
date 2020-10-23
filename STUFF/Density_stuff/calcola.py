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
alpha = 0.44 # spectral index
bol_corr = 3.8

k_corr = alpha-1

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
'''
num_sources = len(open('sdss_in_class.txt').readlines(  ))
z_sdss					 = np.zeros(num_sources-1)
mag 			 = np.zeros(num_sources-1)
with open('sdss_in_class.txt') as fp:
	next(fp)
	for i in range(num_sources-1):
		string = fp.readline().split()
		if (float(string[6]) > 9) & (float(string[2]) > 3):
			mag[i] 				= float(string[2])
			z_sdss[i]			= float(string[4])

z = np.arange(0.5, 5, 0.01)
Dl= np.zeros(len(z))
for i in range(len(z)):
	Dl[i] = ned_calc(z[i], H0, Omega_m, Omega_vac)
m_AB = -2.5*( math.log10(1.2) - math.log10(3.8) + 46 +0.56*np.log10(1+z) - 0.44*math.log10(6580/1400) + math.log10(0.000000140/299792458) - math.log10(4*math.pi*3.09) -2*np.log10(Dl) -48 ) - 48.6
plt.plot(z, -m_AB)
plt.scatter(z_sdss, -mag, c='r', s=0.5)
plt.ylim(-22.5, -15)
plt.show()'''

'''lambd = np.arange(3000, 7500, 1)
f_lambd = 1.62*10**-11*(lambd**(-1.56) + 0.00001*np.exp(-(lambd - 1215)**2/(100)) + 0.000001*np.exp(-(lambd - 6600)**2/(1000)))
nu = 2997924580000000000/lambd
f_nu = f_lambd*(lambd**2)*3.34*10**4*10**-23
f_1400 = f_lambd*(lambd/1400)**(+1.56)
f_nuxnu = f_nu*nu
f_nuxnu_1400 = f_nuxnu*(1400/lambd)**(-0.56)
plt.plot(-nu, f_nu)
plt.show()

z = 4.5

f_nu_rf = f_nu/(1+z)
nu_rf = nu*(1+z)
f_lambd_rf = f_lambd*(1+z)
lambd_rf = lambd/(1+z)
f_nuxnu_rf = f_nu_rf*nu_rf
f_lambdxlambd_rf = f_lambd_rf*lambd_rf

#plt.plot(nu_rf, f_nuxnu_rf)
#plt.show()

f_nu_k = f_nu*(1+z)**(k_corr)

#plt.subplot(2,1,1)
#plt.plot(f_lambd)
#plt.xlim(0, 3e15)
#plt.ylim(0, 1e-28)
#plt.subplot(2,1,2)
#plt.plot(nu_rf, f_nu_rf)
#plt.xlim(0, 3e15)
#plt.ylim(0, 1e-28)
plt.show()'''

def from_m_to_f(m):
	f = -(m+48.6)/2.5
	return f
	
def from_logf_to_m(f):
	m = -2.5*f-48.6
	return m
	
def from_m_to_L(m, lambda_obs, lambda_rf, z):
	f_obs = from_m_to_f(m)
	distz = np.zeros(len(z))
	for i in range(len(z)):
		distz[i] = ned_calc(z[i], H0, Omega_m, Omega_vac)
	return f_obs + np.log10((1+z)**(k_corr)*(lambda_obs/lambda_rf)**(-0.44)*4*math.pi) + 2*np.log10(distz*3.09) + 48 + np.log10(299792458000000000/lambda_rf)
	
def from_L_to_m(L, lambda_obs, lambda_rf, z):
	distz = np.zeros(len(z))
	for i in range(len(z)):
		distz[i] = ned_calc(z[i], H0, Omega_m, Omega_vac)
	f_obs = L/(4*math.pi*(distz*3.09*10**24)**2*(1+z)**(k_corr)*(lambda_obs/lambda_rf)**(-0.44))
	return from_logf_to_m(math.log10(f_obs))

taglio = -1 + 9 + 38 + math.log10(1.26/3.8) 
zmin = 0.75
zmax = 3.75
bin_width = 1
z_bin = np.arange(zmin, zmax+bin_width, bin_width) 	# array contenente z_min e z_max di ogni bin

for j in range(len(z_bin)-1):
	name 			 = []
	z					 = []
	z_SDSS		 = []
	radio_flux = []
	flag_class = []
	mag 			 = []
	m_shen 		 = []
	edd_ratio  = []
	dist_z 		 = []
	
	with open('sdss_in_class1.txt') as fp:
		next(fp)
		for line in fp:
			string = line.split()
			if (float(string[6]) > 9) & (float(string[4]) > z_bin[j]) & (float(string[4]) <= z_bin[j+1]):
				name.append(string[0])
				radio_flux.append(float(string[1]))
				mag.append(float(string[2]))
				flag_class.append(float(string[3]))
				z.append(float(string[4]))
				z_SDSS.append(float(string[5]))
				m_shen.append(float(string[6]))
				edd_ratio.append(float(string[8]))
				dist_z.append(ned_calc(float(string[4]), H0, Omega_m, Omega_vac))
					
	name.append(string[0])
	radio_flux = np.asarray(radio_flux)
	mag = np.asarray(mag)
	flag_class = np.asarray(flag_class)
	z = np.asarray(z)
	z_SDSS = np.asarray(z_SDSS)
	m_shen = np.asarray(m_shen)
	edd_ratio = np.asarray(edd_ratio)			
	dist_z = np.asarray(dist_z)
	L = -(mag+48.6)/2.5 + k_corr*np.log10(1+z) -0.44*np.log10(6580/1400) + np.log10(4*math.pi) + 2*np.log10(dist_z*3.9) + 48 + np.log10(299792458/0.000000140) 
	mag_limit = 21.5 -5*np.log10(41200/dist_z)-2.5*k_corr*np.log10((1+4.5)/(1+z))	
	#edd = [elem + m + 38 + math.log10(1.26) for elem,m in zip(edd_ratio,m_shen) if elem != 0]
	#L = from_m_to_L(mag, 6160, 1400, z)
	persi = [elem for elem in L if elem <= taglio]
	perc_persi = len(persi)/len(mag)
	#edd = np.asarray(edd)

	fig, ax = plt.subplots()
	plt.xlim(45, 48)
	#plt.plot(z, mag_limit)
	plt.ylim(0, 100)
	ax.hist(L)
	plt.axvline(x=taglio, linestyle=':', color='red')	
	fig.text(0.1, 0.8, 'Persi = {0:.1f}%'.format(perc_persi*100), fontsize=12, transform=ax.transAxes)
	plt.title(r"$L\lambda1400$ " + "z = {0}-{1}".format(z_bin[j], z_bin[j+1]))
	#plt.ylim(0, 70)
	plt.savefig("L1400{}.png".format(j))