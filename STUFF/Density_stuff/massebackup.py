#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import statistics 
import math

import pylab as P
from scipy import stats

def draw_hist(truevalue, data):

	sLim=0.68 #Intervallo di confidenza
	nBins=int(Ntries/50)
	P.hist(data,bins=nBins,alpha=0.5, color='g')
	P.xlabel(r'Mass ($M_{\odot}$)')

	# Let's find the values at the lower- and upper- "sLim" bounds:
	Med = truevalue # np.median(data)
	gHi = np.where(data >= Med)[0]
	gLo = np.where(data < Med)[0]

	vSortLo=np.sort(data[gLo])
	vSortHi=np.sort(data[gHi])

	NormLo = vSortLo[np.int((1.0-sLim)*np.size(vSortLo))]
	NormHi = vSortHi[np.int(sLim      *np.size(vSortHi))]

	## Let's take a look - how do those limits look on the histogram?
	P.plot([Med, Med],[1,Ntries/12], 'k--', lw=1)
	P.plot([NormLo, NormLo],[1,Ntries/12], 'k-.', lw=1)
	P.plot([NormHi, NormHi],[1,Ntries/12], 'k-.', lw=1)

	# Do some annotations on the plot with these limits:
	P.annotate('%i percent limits' % (sLim*100), (0.6,0.9), xycoords='axes fraction')
	P.title('Mass: <%.2f> -%.2f +%.2f' % (Med, Med-NormLo, NormHi-Med))

	return Med-NormLo, NormHi-Med

def random_asy_distr(media, errl, errh):
	norm = np.sqrt(-errl/errh)
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
	print(np.size(np.concatenate((dlow, dhigh))))
	return np.concatenate((dlow, dhigh))
	
masse_sigma = np.zeros(19)
masse_fwhm = np.zeros(19)
name = []
z = np.zeros(19)
sigma = np.zeros(19)
errsigma = np.zeros(19)
fwhm = np.zeros(19)
errfwhm = np.zeros(19)
lum1350 = np.zeros(19)
errlum1350h = np.zeros(19)
errlum1350l = np.zeros(19)

lum_file = open("daticompllog.txt", 'r')
lines = lum_file.readlines()[2:]
for i, line in enumerate(lines):
	lum1350[i] = float(line.split()[7])
	errlum1350h[i] = float(line.split()[8])
	errlum1350l[i] = float(line.split()[9])
lum_file.close()

uncert_file = open("uncertainties.txt", 'r')
lines = uncert_file.readlines()[1:]
for i, line in enumerate(lines):
	name.append(line.split()[0])
	z[i] = float(line.split()[1])
	sigma[i] = float(line.split()[2])
	errsigma[i] = float(line.split()[3])
	fwhm[i] = float(line.split()[4])
	errfwhm[i] = float(line.split()[5])
	masse_sigma[i] = 6.73 + 2*math.log10(sigma[i]/1000) + 0.53*(lum1350[i]-44)
	masse_fwhm[i] = 6.66 + 2*math.log10(fwhm[i]/1000) + 0.53*(lum1350[i]-44)
uncert_file.close()

######################################### CALCOLO ERRORI MASSE #################################################

Ntries = 2000
mockMsigma = np.zeros(Ntries)
mockMfwhm = np.zeros(Ntries)
Msigma_err = np.zeros([19,2])
Mfwhm_err = np.zeros([19,2])

for i in range(19):
	randomsigma = np.random.normal(sigma[i], errsigma[i], mockMsigma.shape)
	randomfwhm  = np.random.normal(fwhm[i], errfwhm[i], mockMfwhm.shape)
	randomlum   = random_asy_distr(lum1350[i],errlum1350l[i],errlum1350h[i])
	mockMsigma = 6.73 + 2*np.log10(randomsigma/1000) + 0.53*(randomlum-44)
	mockMfwhm  = 6.66 + 2*np.log10(randomfwhm/1000)  + 0.53*(randomlum-44)
	Msigma_err[i][0], Msigma_err[i][1] = draw_hist(masse_sigma[i], mockMsigma)
	P.savefig("{}sigma".format(name[i]))
	P.cla()
	Mfwhm_err[i][0], Mfwhm_err[i][1] = draw_hist(masse_fwhm[i], mockMfwhm)
	P.savefig("{}fwhm".format(name[i]))
	P.cla()

####### Scrittura su file ########

output = open("masse_python.txt",'w+')
output.write("name\tz\tsigma(km/s)\terr_sigma\tFWHM(km/s)\terr_FWHM\tloglum1350\terr_lum-\terr_lum+\tlogM_sigma\terr_M-\terr_M+\tlogM_FWHM\terr_M-\terr_M+\n")
for i in range(19):
	output.write("{0}\t{1:.2f}\t{2:.0f}\t{3:.0f}\t{4:.0f}\t{5:.0f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\t{8:.2f}\t{9:.2f}\t{10:.2f}\t{11:.2f}\t{12:.2f}\t{13:.2f}\n".format(name[i],z[i],sigma[i],errsigma[i],fwhm[i],errfwhm[i],lum1350[i],errlum1350l[i],errlum1350h[i],masse_sigma[i],Msigma_err[i][0],Msigma_err[i][1],masse_fwhm[i],Mfwhm_err[i][0],Mfwhm_err[i][1]))
output.close()
x = np.arange(7.0, 11.5, 0.1)
y = x
plt.plot(x, y, linewidth=1)
plt.errorbar(masse_sigma, masse_fwhm, xerr=Msigma_err.T, yerr=Mfwhm_err.T, fmt='ko', markersize=1)
plt.xlim([8.0, 10.5])
plt.ylim([8.0, 10.5])
plt.xlabel(r"$M_{\sigma}$")
plt.ylabel(r"$M_{FWHM}$")
plt.savefig("PyMass_fwhmVsSigma.eps")
plt.cla()
	