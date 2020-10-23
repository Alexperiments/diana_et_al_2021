#!/usr/bin/python3
from functions import *
from scipy.odr import Model, RealData, ODR
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator

def plotM_eps(parx, pary, errx=np.zeros(19), erry=np.zeros(19), xlabel=None, ylabel=None, title=None, x=np.arange(7.0, 11.5, 0.1), y=np.arange(7.0, 11.5, 0.1), xlim=None, ylim=None, dpi=300):
    plt.close('all')
    fig, ax = plt.subplots()
    ax.errorbar(parx, pary, yerr=erry.T, xerr=errx.T, c='black', marker='D', fmt='o', markersize=5, elinewidth=1, capsize=3)
    ax.plot(x,y, c='dodgerblue', lw=2)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(  AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(  AutoMinorLocator(5))
    ax.xaxis.set_tick_params(which='major', direction='inout', length=10)
    ax.yaxis.set_tick_params(which='major', direction='inout', length=10)
    ax.xaxis.set_tick_params(which='minor', direction='inout', length=5)
    ax.yaxis.set_tick_params(which='minor', direction='inout', length=5)
    for i in range(len(parx)):
        plt.text(parx[i], pary[i]+0.1, name[i], rotation=45, fontsize=4)
    ax.set_xticks(np.arange(round(min(parx),2), round(max(parx)+(max(parx)-min(parx))/10, 2), round((max(parx)-min(parx))/10, 2)))
    ax.set_yticks(np.arange(round(min(pary), 2), round(max(pary)+(max(pary)-min(pary))/10, 2), round((max(pary)-min(pary))/10, 3)))
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(linestyle='-.', lw=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig("./output/" + title ,dpi=dpi)
    plt.clf()

def plot_hist(data1, label1, title):
    plt.close('all')
    fig, ax = plt.subplots()
    ax.hist(data1, bins='freedman', alpha=0.5, label=label1)
    ax.legend(loc='upper right')
    plt.title(title)
    plt.savefig(title + ".png", dpi=300)
    plt.clf()

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
errlum1350 = np.zeros([19,2])
radio_flux = np.zeros(19)
line_lum = np.zeros(19)
errline_lum = np.zeros([19,2])
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

def quad_func(p, x):
     m, c = p
     return m*x + c

with open("./output/blazars_data.txt", 'r') as fp:
    lines = fp.readlines()[1:]
    for i, line in enumerate(lines):
        name.append(line.split()[0])
        z[i] = float(line.split()[1])
        sigma[i] = float(line.split()[2])
        errsigma[i] = float(line.split()[3])
        fwhm[i] = float(line.split()[4])
        errfwhm[i] = float(line.split()[5])
        EW[i] = float(line.split()[6])
        errEW[i] = float(line.split()[7])
        radio_flux[i] = float(line.split()[8])
        lum1350[i] = float(line.split()[9])
        errlum1350[i][0] = float(line.split()[10])
        errlum1350[i][1] = -float(line.split()[11])
        line_lum[i] = float(line.split()[12])
        errline_lum[i][0] = float(line.split()[13])
        errline_lum[i][1] = -float(line.split()[13])
        masse_sigma[i] = float(line.split()[14])
        Msigma_err[i][0] = float(line.split()[15])
        Msigma_err[i][1] = -float(line.split()[16])
        masse_fwhm[i] = float(line.split()[17])
        Mfwhm_err[i][0] = float(line.split()[18])
        Mfwhm_err[i][1] = -float(line.split()[19])
        eddratio_sigma[i] = float(line.split()[20])
        eddratio_fwhm[i] = float(line.split()[21])
        blueshift[i] = float(line.split()[22])
        errblueshift[i] = float(line.split()[23])
        blueshift_coat[i] = float(line.split()[24])
        errblueshift_coat[i] = float(line.split()[25])
        masse_fwhm_denney[i] = float(line.split()[26])
        Mfwhm_err_denney[i][0] = -float(line.split()[27])
        Mfwhm_err_denney[i][1] = +float(line.split()[28])
        masse_fwhm_coatman[i] = float(line.split()[29])
        Mfwhm_err_coatman[i][0] = -float(line.split()[30])
        Mfwhm_err_coatman[i][1] = +float(line.split()[31])
        M_bontà[i] = float(line.split()[32])
        M_bontà_err[i][0] = -float(line.split()[33])
        M_bontà_err[i][1] = float(line.split()[34])
        M_disk[i] = float(line.split()[35])
        M_disk_err[i][0] = -float(line.split()[36])
        M_disk_err[i][1] = float(line.split()[37])
        Mfwhm_linea[i] = float(line.split()[38])
        Mfwhm_linea_err[i][0] = -float(line.split()[39])
        Mfwhm_linea_err[i][1] = float(line.split()[40])

z_bin = np.linspace(3.9, 5.5, 25)
dl_bin = np.zeros(len(z_bin))

for i in range(len(z_bin)):
    dl_bin[i] = ned_calc(z_bin[i])

k_bol = 2.5
logeddr_limit = -0.5

k_M_limit = math.log10(k_bol) -17.12 + math.log10(1300) + np.log10(1+z_bin) + math.log10(4*math.pi*3.09*3.09) + 48 + 2*np.log10(dl_bin) - math.log10(1.26) - 38  - logeddr_limit  #Limite per mag=21 e edd_ratio = 0.3

lum1350, line_lum = zip(*sorted(zip(lum1350, line_lum)))

quad_model = Model(quad_func)
data = RealData(lum1350, line_lum, sx=-errlum1350[:][0], sy=errline_lum[:][0])

odr = ODR(data, quad_model, beta0=[1.1, -6.7])

out = odr.run()

x_fit = np.linspace(45.5, 47.5, 1000)
y_fit = quad_func(out.beta, x_fit)

plotM_eps(lum1350, line_lum, errlum1350, errline_lum, xlabel= r"$log(\lambda L_{1350})(erg s^{-1})$", ylabel=r"$logL_{line}(erg s^{-1})$",  title="Lline_L1350.eps", x=x_fit, y=y_fit, xlim=[45.5, 47.5])

plotM_eps(masse_sigma, eddratio_sigma, Msigma_err , xlabel= r"$logM_{\sigma}(M_{\odot})$", ylabel=r"$log(\lambda_{edd})$",  title="eddr_Msigma.eps", x=np.linspace(8.0, 10.5, 100), y=8.64 - np.linspace(8.0, 10.5, 100), xlim=[8.0, 10.5])

plotM_eps(z, masse_sigma, erry=Msigma_err, xlabel="z", ylabel=r"$logM_{\sigma}(M_{\odot})$", title="MassLimitsigma.eps", x=z_bin, y=k_M_limit, xlim=[z_bin[0], z_bin[-1]], ylim=[min(masse_sigma)-0.2,max(masse_sigma)+0.2])

plotM_eps(z, masse_fwhm, erry=Mfwhm_err, xlabel="z", ylabel=r"$logM_{FWHM}(M_{\odot})$", title="MassLimitfwhm.eps", x=z_bin, y=k_M_limit, xlim=[z_bin[0], z_bin[-1]], ylim=[min(masse_fwhm)-0.2,max(masse_fwhm)+0.2])

plotM_eps(masse_fwhm, Mfwhm_linea, Mfwhm_err, Mfwhm_linea_err, r"$M_{1350}(M_{\odot})$", r"$M_{CIV}(M_{\odot})$", "MRigaVSfwhm.eps", xlim=[8.0, 10.5], ylim=[8.0, 10.5])

plotM_eps(masse_sigma, Mfwhm_linea, Msigma_err, Mfwhm_linea_err, r"$M_{1350}(M_{\odot})$", r"$M_{CIV}(M_{\odot})$", "MRigaVSsigma.eps", xlim=[8.0, 10.5], ylim=[8.0, 10.5])

plotM_eps(masse_fwhm, M_bontà, Mfwhm_err, M_bontà_err, r"$M_{FWHM}(M_{\odot}) (VP06)$", r"$M_{\sigma}(M_{\odot}) (Dalla Bontà)$", "BontaVSfwhm.eps", xlim=[8.0, 10.8], ylim=[8.0, 10.8])

plotM_eps(masse_sigma, M_bontà, Msigma_err, M_bontà_err, r"$M_{\sigma}(M_{\odot}) (VP06)$", r"$M_{\sigma}(M_{\odot}) (Dalla Bontà)$", "BontaVSsigma.eps", xlim=[8.0, 10.8], ylim=[8.0, 10.8])

plotM_eps(masse_sigma, M_disk, Msigma_err, M_disk_err, r"$M_{\sigma}(M_{\odot})$", r"$M_{disk}(M_{\odot})$", "sigmaVSdisk.eps", xlim=[8.0, 10.8], ylim=[8.0, 10.8])

plotM_eps(masse_fwhm, M_disk, Mfwhm_err, M_disk_err, r"$M_{FWHM}(M_{\odot})$", r"$M_{disk}(M_{\odot})$", "fwhmVSdisk.eps", xlim=[8.0, 10.8], ylim=[8.0, 10.8])

plotM_eps(M_bontà, M_disk, M_bontà_err, M_disk_err, r"$M_{\sigma}(M_{\odot}) (Bontà)$", r"$M_{disk}(M_{\odot})$", "BontaVSdisk.eps", xlim=[8.0, 10.8], ylim=[8.0, 10.8])

plotM_eps(M_bontà, Mfwhm_linea, M_bontà_err, Mfwhm_linea_err, r"$M_{\sigma}(M_{\odot}) (Bontà)$", r"$M_{CIV}(M_{\odot})$", "BontaVSlinea.eps", xlim=[8.0, 10.8], ylim=[8.0, 10.8])

plotM_eps(Mfwhm_linea, M_disk, Mfwhm_linea_err, M_disk_err, r"$M_{CIV}(M_{\odot})$", r"$M_{disk}(M_{\odot})$", "MRigaVSDisk.eps", xlim=[8.0, 10.5], ylim=[8.0, 10.5])

plotM_eps(EW, Mfwhm_linea-masse_sigma, errEW, Mfwhm_linea_err, r"$EW(\AA)$", r"$M_{CIV} - M_{\sigma}(M_{\odot})$", "Linea-sigmaVSEW.eps", x=0, y=0)

plotM_eps(EW, M_disk-masse_sigma, errEW, M_disk_err, r"$EW(\AA)$", r"$M_{disk} - M_{\sigma}(M_{\odot})$", "Disk-sigmaVSEW.eps", x=0, y=0)

xlum1350 = np.linspace(40, 47.5, 1000)
ylumCIV = 7.66 + 0.863*xlum1350
plt.close('all')
fig, ax = plt.subplots()
ax.errorbar(line_lum, lum1350, xerr=errline_lum.T, yerr=errlum1350.T, c='black', marker='D', fmt='o', markersize=5, elinewidth=1, capsize=3)
ax.plot(xlum1350,ylumCIV, c='dodgerblue', lw=2)
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_locator(  AutoMinorLocator(5))
ax.xaxis.set_minor_locator(  AutoMinorLocator(5))
ax.xaxis.set_tick_params(which='major', direction='inout', length=10)
ax.yaxis.set_tick_params(which='major', direction='inout', length=10)
ax.xaxis.set_tick_params(which='minor', direction='inout', length=5)
ax.yaxis.set_tick_params(which='minor', direction='inout', length=5)
ax.set_yticks(np.arange(round(min(lum1350),2), round(max(lum1350)+(max(lum1350)-min(lum1350))/10, 2), round((max(lum1350)-min(lum1350))/10, 2)))
ax.set_xticks(np.arange(round(min(line_lum), 2), round(max(line_lum)+(max(line_lum)-min(line_lum))/10, 2), round((max(line_lum)-min(line_lum))/10, 3)))
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
ax.yaxis.set_major_locator(plt.MaxNLocator(10))
plt.ylim([min(lum1350)-0.2,max(lum1350)+0.2])
plt.xlim([min(line_lum)-0.2,max(line_lum)+0.2])
plt.grid(linestyle='-.', lw=0.5)
ax.fill_between(xlum1350, ylumCIV-0.2, ylumCIV+0.2, facecolor='grey', alpha=0.5)
plt.xlabel(r"$L_{CIV} (erg s^{-1})$")
plt.ylabel(r"$L\lambda1350 (erg s^{-1})$")
plt.savefig("./output/" + "lineVScontinuumSHEN.eps" ,dpi=300)
plt.clf()

def K_bol_calc(L1400):
    ()
    return 3.81#7*(10**(L1400-42))**(-0.1)
