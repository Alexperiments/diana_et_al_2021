#!/usr/bin/python3

from functions import *
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator

draw_plots = False

k_bol = 3.81

masse_sigma = np.zeros(19)
masse_fwhm = np.zeros(19)
masse_fwhm_denney = np.zeros(19)
masse_fwhm_coatman =  np.zeros(19)
name = []
z = np.zeros(19)
sigma = np.zeros(19)
errsigma = np.zeros(19)
fwhm = np.zeros(19)
errfwhm = np.zeros(19)
lum1350 = np.zeros(19)
errlum1350h = np.zeros(19)
errlum1350l = np.zeros(19)
radio_flux = np.zeros(19)
line_lum = np.zeros(19)
errline_lum = np.zeros(19)
EW = np.zeros(19)
errEW = np.zeros(19)
blueshift = np.zeros(19)
errblueshift = np.zeros(19)
blueshift_coat = np.zeros(19)
errblueshift_coat = np.zeros(19)
eddratio_fwhm = np.zeros(19)
eddratio_sigma = np.zeros(19)
erredd_fwhmh = np.zeros(19)
erredd_fwhml = np.zeros(19)
erredd_sigmah = np.zeros(19)
erredd_sigmal = np.zeros(19)
Msigma_err = np.zeros([19,2])
Mfwhm_err = np.zeros([19,2])
Mfwhm_err_denney = np.zeros([19,2])
Mfwhm_err_coatman = np.zeros([19,2])
Mfwhm_linea = np.zeros(19)
Mfwhm_linea_err = np.zeros([19,2])
M_bontà = np.zeros(19)
M_bontà_err = np.zeros([19,2])
M_disk = np.zeros(19)
M_disk_err = np.zeros([19,2])
radioloudness = np.zeros(19)

uncert_file = open("uncertainties.txt", 'r')
lines = uncert_file.readlines()[1:]
for i, line in enumerate(lines):
	name.append(line.split()[0])
	z[i] = float(line.split()[1])
	sigma[i] = float(line.split()[2])
	errsigma[i] = float(line.split()[3])
	fwhm[i] = float(line.split()[4])
	errfwhm[i] = float(line.split()[5])
	lum1350[i] = float(line.split()[6])
	errlum1350h[i] = float(line.split()[7])
	errlum1350l[i] = float(line.split()[8])
	radio_flux[i] = float(line.split()[9])
	line_lum[i] = float(line.split()[10])
	errline_lum[i] = float(line.split()[11])
	EW[i] = float(line.split()[12])
	errEW[i] = float(line.split()[13])
	blueshift[i] = float(line.split()[14])
	errblueshift[i] = float(line.split()[15])
	blueshift_coat[i] = float(line.split()[16])
	errblueshift_coat[i] = float(line.split()[17])
uncert_file.close()

with open("./output/blazars_data.txt", 'r') as fp:
	lines = fp.readlines()[1:]
	for i, line in enumerate(lines):
		if calcola_di_nuovo_tutti_dati:
			masse_sigma[i] = 6.73 + 2*math.log10(sigma[i]/1000) + 0.53*(lum1350[i]-44)
			masse_fwhm[i] = 6.66 + 2*math.log10(fwhm[i]/1000) + 0.53*(lum1350[i]-44)
			eddratio_fwhm[i] = np.log10(k_bol) + lum1350[i] - (np.log10(1.26) + 38 + masse_fwhm[i])
			eddratio_sigma[i] = np.log10(k_bol) + lum1350[i] - (np.log10(1.26) + 38 + masse_sigma[i])
		else:
			masse_sigma[i] = float(line.split()[14])
			Msigma_err[i][0] = float(line.split()[15])
			Msigma_err[i][1] = float(line.split()[16])
			masse_fwhm[i] = float(line.split()[17])
			Mfwhm_err[i][0] = float(line.split()[18])
			Mfwhm_err[i][1] = float(line.split()[19])
			eddratio_sigma[i] = float(line.split()[20])
			eddratio_fwhm[i] = float(line.split()[21])
		FWHM_coatman = fwhm[i]/(0.36*blueshift_coat[i]/1000 + 0.62)
		masse_fwhm_denney[i] = masse_fwhm[i] + 0.046 - 2.01*math.log10(fwhm[i]/sigma[i])
		masse_fwhm_coatman[i] = 6.71 + 2*np.log10(FWHM_coatman/1000) + 0.53*(lum1350[i]-44)
		Mfwhm_linea[i] = 7.535 + 0.639*(line_lum[i] - 44) + 0.319*np.log10(fwhm[i])
		M_bontà[i] = math.log10(4.28) + 7.934 + 0.761*(lum1350[i] - 45) + 1.289*(math.log10(sigma[i]) - 3.5)

with open("masse_nowise.txt" ,'r') as fp:
	lines = fp.readlines()[1:]
	for i, line in enumerate(lines):
		M_disk[i] =  float(line.split()[2])
		M_disk_err[i][1] =  float(line.split()[3]) - float(line.split()[2])
		M_disk_err[i][0] =  float(line.split()[4]) - float(line.split()[2])

######################################### CALCOLO ERRORI MASSE #################################################
Ntries = 500
mockMsigma = np.zeros(Ntries)
mockMfwhm = np.zeros(Ntries)
mockDenney = np.zeros(Ntries)
mockCoatman = np.zeros(Ntries)
mockLinea = np.zeros(Ntries)
mockBontà = np.zeros(Ntries)

for i in range(19):
	if calcola_di_nuovo_tutti_dati:
		randomsigma = np.random.normal(sigma[i], errsigma[i], mockMsigma.shape)
		randomfwhm  = np.random.normal(fwhm[i], errfwhm[i], mockMfwhm.shape)
		randomlum   = random_asy_distr(lum1350[i],errlum1350l[i],errlum1350h[i], Ntries)
		mockMsigma = 6.73 + 2*np.log10(randomsigma/1000) + 0.53*(randomlum-44)
		mockMfwhm  = 6.66 + 2*np.log10(randomfwhm/1000)  + 0.53*(randomlum-44)
		#mockDenney = mockMfwhm + 0.219 - 1.63*np.log10(randomfwhm/randomsigma)
		Msigma_err[i][0], Msigma_err[i][1] = draw_hist(masse_sigma[i], mockMsigma)
		plt.savefig("./mass_distributions/"+"{}sigma".format(name[i]), format='png', dpi=300)
		plt.cla()

		Mfwhm_err[i][0], Mfwhm_err[i][1] = draw_hist(masse_fwhm[i], mockMfwhm)
		plt.savefig("./mass_distributions/"+"{}fwhm".format(name[i]), format='png', dpi=300)
		plt.cla()

		randombolcor = random_asy_distr(4,-0.5,1, Ntries)
		mockLbol = np.log10(randombolcor) + randomlum
		mockLedd_fwhm =  mockMfwhm + 38 + np.log10(1.26)
		mockLedd_sigma = mockMsigma + 38 + np.log10(1.26)
		erredd_sigmal[i], erredd_sigmah[i] = draw_hist(np.log10(k_bol) + lum1350[i] - (masse_sigma[i] + 38 + np.log10(1.26)), mockLbol - mockLedd_sigma, False)
		erredd_fwhml[i], erredd_fwhmh[i] = draw_hist(np.log10(k_bol) + lum1350[i] - (masse_fwhm[i] + 38 + np.log10(1.26)), mockLbol - mockLedd_fwhm, False)
		'''erredd_fwhmh[i] = 10**(eddratio_fwhm[i] + erredd_fwhmh[i]) - 10**(eddratio_fwhm[i])
		erredd_fwhml[i] = 10**(eddratio_fwhm[i] - erredd_fwhml[i]) - 10**(eddratio_fwhm[i])
		erredd_sigmah[i] = 10**(eddratio_sigma[i] + erredd_sigmah[i]) - 10**(eddratio_sigma[i])
		erredd_sigmal[i] = 10**(eddratio_sigma[i] - erredd_sigmal[i]) - 10**(eddratio_sigma[i])
		eddratio_fwhm[i] = 10**(eddratio_fwhm[i])
		eddratio_sigma[i] = 10**(eddratio_sigma[i])'''
	else:
		randomsigma = np.random.normal(sigma[i], errsigma[i], mockMsigma.shape)
		randomfwhm  = np.random.normal(fwhm[i], errfwhm[i], mockMfwhm.shape)
		randomlum   = random_asy_distr(lum1350[i],errlum1350l[i],errlum1350h[i], Ntries)
		randomlinea = np.random.normal(line_lum[i], errline_lum[i], mockMfwhm.shape)
		FWHM_coatman = randomfwhm/(0.36*blueshift_coat[i]/1000 + 0.62)
		mockDenney = 6.66 + 2*np.log10(randomfwhm/1000) + 0.53*(randomlum-44) + 0.046 - 2.01*np.log10(randomfwhm/randomsigma)
		mockCoatman  = 6.71 + 2*np.log10(FWHM_coatman/1000)  + 0.53*(randomlum-44)
		mockLinea = 7.535 + 0.639*(randomlinea - 44) + 0.319*np.log10(randomfwhm)
		mockBontà = math.log10(4.28) + 7.934 + 0.761*(randomlum - 45) + 1.289*(np.log10(randomsigma) - 3.5)
		Mfwhm_err_denney[i][0], Mfwhm_err_denney[i][1] = draw_hist(masse_fwhm_denney[i], mockDenney,False)
		Mfwhm_err_coatman[i][0], Mfwhm_err_coatman[i][1] = draw_hist(masse_fwhm_coatman[i], mockCoatman,False)
		Mfwhm_linea_err[i][0], Mfwhm_linea_err[i][1] = draw_hist(Mfwhm_linea[i], mockLinea,False)
		M_bontà_err[i][0], M_bontà_err[i][1] = draw_hist(M_bontà[i], mockBontà,False)
		logS_5GHz_RF = np.log10(radio_flux[i]*(1+z[i])**(0.44-1)) -26
		dl = ned_calc(z[i])
		radioloudness[i] = logS_5GHz_RF - ((lum1350[i] - np.log10(1350) + 2*np.log10(1350) - np.log10(299792458) - 10) - (math.log10(4*math.pi) + 2*np.log10(dl*3.086) + 48))

print(radioloudness)
####### Scrittura su file ########

output = open("./output/blazars_data.txt",'w+')
output.write("name\tz\tsigma(km/s)\terr_sigma\tFWHM(km/s)\terr_FWHM\tEW\terr_EW\tradio_flux\tloglum1350\terr_lum-\terr_lum+\tlogline_lum\terrline_lum\t\tlogM_sigma\terr_M+\terr_M-\tlogM_FWHM\terr_M+\terr_M-\tedd_ratio_sigma\tedd_ratiofwhm\tblueshift\terr_blueshift\tblueshift_Coatman\terr_Coatman\tlogM_Denney\terr_M+\terr_M-\tlogM_Coatman\terr_M+\terr_M-\tM_bontà\terr_M+\terr_M-\tM_disk\terr_M+\terr_M-\tM_linea\terr_M+\terr_M-\n")
for i in range(19):
	output.write("{0}\t{1:.2f}\t{2:.0f}\t{3:.0f}\t{4:.0f}\t{5:.0f}\t{18:.2f}\t{19:.2f}\t{15:.0f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\t{16:.3f}\t{17:.3f}\t{9:.2f}\t{10:.2f}\t{11:.2f}\t{12:.2f}\t{13:.2f}\t{14:.2f}\t{20:.2f}\t{21:.2f}\t{22:.0f}\t{23:.0f}\t{24:.0f}\t{25:.0f}\t{26:.2f}\t{27:.2f}\t{28:.2f}\t{29:.2f}\t{30:.2f}\t{31:.2f}\t{32:.2f}\t{33:.2f}\t{34:.2f}\t{35:.2f}\t{36:.2f}\t{37:.2f}\t{38:.2f}\t{39:.2f}\t{40:.2f}\n".format(name[i],z[i],sigma[i],errsigma[i],fwhm[i],errfwhm[i],lum1350[i],errlum1350h[i],errlum1350l[i],masse_sigma[i],Msigma_err[i][0],Msigma_err[i][1], masse_fwhm[i], Mfwhm_err[i][0], Mfwhm_err[i][1], radio_flux[i], line_lum[i], errline_lum[i], EW[i], errEW[i], eddratio_sigma[i] , eddratio_fwhm[i], blueshift[i], errblueshift[i], blueshift_coat[i], errblueshift_coat[i], masse_fwhm_denney[i], Mfwhm_err_denney[i][0], -Mfwhm_err_denney[i][1], masse_fwhm_coatman[i], Mfwhm_err_coatman[i][0], -Mfwhm_err_coatman[i][1], M_bontà[i], +M_bontà_err[i][0], -M_bontà_err[i][1], M_disk[i], M_disk_err[i][0], M_disk_err[i][1], Mfwhm_linea[i], Mfwhm_linea_err[i][0], -Mfwhm_linea_err[i][1]))
output.close()

output = open("./output/blazars_Latex.txt",'w+')
output.write(r"Source & $z$ & $\sigma_{\lambda}$ &FWHM & EW & Blueshift & $S_{5GHz}$ & $L\lambda1350$ & $L_{CIV}$ & $logM_{sigma}$ & $logM_{FWHM}$ & $\lambda_{edd} - \sigma_l$ & $\lambda_{edd}$ - FWHM\\ ")
output.write("\n\n")
output.write(r" & &($kms^{-1}$) &($kms^{-1}$) &(\AA) & ($km s^{-1}$) &(mJy) &($erg s^{-1}$) &($erg s^{-1}$) &($M_\odot$) &($M_\odot$) & &\\ ")
output.write("\n\n")
for i in range(19):
    output.write("\hline\n")
    output.write(r"{0}& ${1:.2f}$ &${2:.0f}\pm{3:.0f}$ &${4:.0f}\pm{5:.0f}$ &${15:.1f}\pm{16:.1f}$ &${20:.0f}\pm{21:.0f}$  &${17:.0f}$ &$ {6:.2f}\substack{{+{7:.2f}\\ {8:.2f} }}$ &${18:.3f}\pm{19:.3f}$  &$ {9:.2f}\substack{{+{10:.2f}\\ {11:.2f} }}$ &$ {12:.2f}\substack{{+{13:.2f}\\ {14:.2f} }}$ &$ {22:.2f}\substack{{+{23:.2f}\\ {24:.2f} }}$ &$ {25:.2f}\substack{{+{26:.2f}\\ {27:.2f} }}$ \\".format(name[i], z[i], sigma[i], errsigma[i], fwhm[i], errfwhm[i] ,lum1350[i], errlum1350h[i], errlum1350l[i], masse_sigma[i], Msigma_err[i][1], -Msigma_err[i][0], masse_fwhm[i], Mfwhm_err[i][1], -Mfwhm_err[i][0], EW[i], errEW[i], radio_flux[i], line_lum[i], errline_lum[i], blueshift[i], errblueshift[i], eddratio_sigma[i], erredd_sigmal[i], erredd_sigmah[i], eddratio_fwhm[i], erredd_fwhml[i], erredd_fwhmh[i]))
    output.write("\n")
output.close()

if draw_plots:
	x = np.arange(7.0, 11.5, 0.1)
	y = x
	plt.plot(x, y, linewidth=1)
	plt.errorbar(masse_sigma, masse_fwhm, xerr=Msigma_err.T, yerr=Mfwhm_err.T, fmt='ko', markersize=1.5)
	plt.xlim([8.0, 10.5])
	plt.ylim([8.0, 10.5])
	plt.xlabel(r"$M_{\sigma}$")
	plt.ylabel(r"$M_{FWHM}$")
	plt.savefig("./output/fwhmVsSigma.eps")
	plt.clf()

	plt.errorbar(eddratio_fwhm, blueshift, xerr=[erredd_fwhml, erredd_fwhmh], yerr=errblueshift, fmt='ko', elinewidth=0.4, markersize=1.5)
	plt.xlabel(r"$Log \lambda_{edd} (FWHM)$")
	plt.ylabel(r"Blueshift (km/s)")
	plt.savefig("./output/eddBlueshiftFWHM.eps")
	plt.clf()

	plt.errorbar(eddratio_sigma, blueshift, xerr=[erredd_sigmal, erredd_sigmah], yerr=errblueshift, fmt='ko', elinewidth=0.4, markersize=1.5)
	plt.xlabel(r"$Log \lambda_{edd} (\sigma)$")
	plt.ylabel(r"Blueshift (km/s)")
	plt.savefig("./output/eddBlueshiftsigma.eps")
	plt.clf()

	xmasses = np.arange(8.0, 11, 0.1)
	ymasses = 8.5 - xmasses

	plt.errorbar(masse_sigma, eddratio_sigma, xerr=Msigma_err.T, yerr=[erredd_sigmal, erredd_sigmah], fmt='ko', elinewidth=0.4, markersize=1.5)
	plt.plot(xmasses,ymasses)
	plt.xlabel(r"$Log M_{BH} - \sigma (M_{\odot})$")
	plt.ylabel(r"$Log \lambda_{edd} (\sigma)$")
	plt.savefig("./output/eddrVSM_sigma.eps")
	plt.clf()

	plt.errorbar(masse_fwhm, eddratio_fwhm, xerr=Mfwhm_err.T, yerr=[erredd_fwhml, erredd_fwhmh], fmt='ko', elinewidth=0.4, markersize=1.5)
	plt.plot(xmasses,ymasses)
	plt.xlabel(r"$Log M_{BH} - FWHM (M_{\odot})$")
	plt.ylabel(r"$Log \lambda_{edd} (\sigma)$")
	plt.savefig("./output/eddrVSM_fwhm.eps")
	plt.clf()

	plt.errorbar(eddratio_fwhm, fwhm, xerr=[erredd_fwhml, erredd_fwhmh], yerr=errfwhm, fmt='ko', elinewidth=0.4, markersize=1.5)
	plt.ylabel(r"FWHM (km/s)")
	plt.xlabel(r"$Log \lambda_{edd} (FWHM)$")
	plt.savefig("./output/eddrVSfwhm.eps")
	plt.clf()

	plt.errorbar(eddratio_sigma, sigma, xerr=[erredd_sigmal, erredd_sigmah], yerr=errsigma, fmt='ko', elinewidth=0.4, markersize=1.5)
	plt.ylabel(r"$\sigma (km/s)$")
	plt.xlabel(r"$Log \lambda_{edd} (FWHM)$")
	plt.savefig("./output/eddrVSsigma.eps")
	plt.clf()

x = np.arange(7.0, 11.5, 0.1)
y = x
plt.subplot(221)
plt.plot(x, y, linewidth=1)
plt.errorbar(masse_sigma, masse_fwhm, xerr=Msigma_err.T, yerr=Mfwhm_err.T, c='black', marker='D', fmt='o', markersize=1.5, elinewidth=0.8, capsize=2)
plt.xlim([8.0, 10.5])
plt.ylim([8.0, 10.5])
plt.ylabel(r"$M_{FWHM}$")
plt.title("Non Corretto")

plt.subplot(222)
plt.plot(x, y, linewidth=1)
plt.errorbar(masse_sigma, masse_fwhm_denney, xerr=Msigma_err.T, yerr=Mfwhm_err_denney.T, c='green', marker='D', fmt='o', markersize=1.5, elinewidth=0.8, capsize=2)
plt.xlim([8.0, 10.5])
plt.ylim([8.0, 10.5])
plt.title("Denney")

plt.subplot(223)
plt.plot(x, y, linewidth=1)
plt.errorbar(masse_sigma, masse_fwhm_coatman, xerr=Msigma_err.T, yerr=Mfwhm_err_coatman.T, c='red', marker='D', fmt='o', markersize=1.5, elinewidth=0.8, capsize=2)
plt.xlim([8.0, 10.5])
plt.ylim([8.0, 10.5])
plt.xlabel(r"$M_{\sigma}$")
plt.ylabel(r"$M_{\sigma}$")
plt.title("Coatman")

plt.subplot(224)
plt.plot(x, y, linewidth=1)
plt.errorbar(masse_sigma, Mfwhm_linea, xerr=Msigma_err.T, yerr=Mfwhm_linea_err.T, c='blue', marker='D', fmt='o', markersize=1.5, elinewidth=0.8, capsize=2)
plt.xlim([8.0, 10.5])
plt.ylim([8.0, 10.5])
plt.xlabel(r"$M_{\sigma}$")
plt.title(r"$L_{line}$")
plt.savefig("./output/fwhmVsSigma.eps")
plt.clf()
