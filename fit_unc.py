#!/usr/bin/python3
from functions import *

Ntries = 20
best_second_moment = np.zeros(19)
sigmaerr = np.zeros(19)
best_fwhm = np.zeros(19)
fwhmerr = np.zeros(19)
l1350 = np.zeros(19)
l1350errh = np.zeros(19)
l1350errl = np.zeros(19)
EW = np.zeros(19)
EW_err = np.zeros(19)
radio_flux = np.zeros(19)
CIV_lum = np.zeros(19)
CIV_lum_err = np.zeros(19)
noise = np.zeros(19)
flux1350 = np.zeros(19)
StoN = np.zeros(19)
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

for i in range(19):
####################### READ INPUT FILE #######################
    to_be_fit_file = r'./CIV/'+names[i]+ ".d"

    with open(to_be_fit_file) as fp:
        lines = fp.readlines()[2:]

    x = np.zeros(len(lines))
    y = np.zeros(len(lines))
    for l,line in enumerate(lines):
        x[l] = float(line.split()[0])
        y[l] = float(line.split()[1])

####################### ESTIMATE L135O #######################

    x_unity = np.arange(1350,1750,0.1)
    x = np.asarray(x)

    with open("spettri_txt/"+names[i]+".txt") as fp:
	    lines = fp.readlines()[2:]

    wl = np.zeros(len(lines))
    fl = np.zeros(len(lines))
    for l,line in enumerate(lines):
        wl[l] = float(line.split()[0])
        fl[l] = float(line.split()[1])
    wl=np.asarray(wl)/(1+z[i])
    fl=np.asarray(fl)*(1+z[i])

    m,q = continuum_estimate(np.vstack((wl, fl)), bg_int[i])
    ycontinuum = m*x+q
    y = y - ycontinuum

    flwind = [fl[k] for k, elem in enumerate(wl) if ((elem > 1340) and (elem < 1360))]
    dl = ned_calc(z[i])

    flux1350[i] = statistics.mean(flwind)
    l1350[i] = math.log10(flux1350[i]) - 17 + math.log10(4*math.pi) + 2*math.log10(dl*3.086) + 48 + math.log10(1350)
    l1350errh[i] = math.log10(1 + statistics.stdev(flwind)/flux1350[i])
    l1350errl[i] = math.log10(1 - statistics.stdev(flwind)/flux1350[i])

####################### FIT CIV AND ESTIMATE EW #######################
    altezza_linea = m*1549+q

    ym, ycalc, integral, centroid, best_second_moment[i], blueshift[i] = fit_CIV(names[i], x, y)
    EW[i] = integral/altezza_linea
    blueshift_coat[i] = 299792.458*(1549-centroid)/1549

####################### ESTIMATE FIT PARAMETERS #######################

    max_y = max(ym)  # Find the maximum y value
    xs = [l for k,l in enumerate(x) if ym[k] > max_y/2.0]
    best_fwhm[i] = (max(xs) - min(xs))*299792/1549 # Print the points at half-maximum

####################### CIV LUMINOSITY #######################

    CIV_lum[i] = math.log10(integral) - 17 + math.log10(4*math.pi) + 2*math.log10(dl*3.086) + 48

####################### READING RADIO FLUX #######################

    with open("sdss_in_class.txt", "r") as input:
        for li in input.readlines()[1:]:
            if li.split()[0] == names[i]:
                radio_flux[i] = float(li.split()[1])
                break

    if flag_calcola_errori:
        continuum = y - ym
        noise[i] = statistics.stdev(continuum[np.where(x > 1510)[0][0]:np.where(x < 1590)[0][-1]])
        second_moment = np.zeros(Ntries)
        fwhm = np.zeros(Ntries)
        mock_EW = np.zeros(Ntries)
        mock_Lciv = np.zeros(Ntries)
        mock_blueshift = np.zeros(Ntries)
        mock_blueshift_coat = np.zeros(Ntries)
        print('processing '+names[i]+'...')
        for j in range(Ntries):
            ymock = ym + np.random.normal(0., 0.75*noise[i], x.shape)
            _, mock_model, integral, mock_centroid, second_moment[j], mock_blueshift[j] = fit_CIV(names[i], x, ymock)

            max_y = max(mock_model)
            xs = [l for k,l in enumerate(x_unity) if mock_model[k] > max_y/2.0]
            fwhm[j] = (max(xs) - min(xs))*299792/1549
            mock_EW[j] = integral/altezza_linea
            mock_Lciv[j] = math.log10(integral) - 17 + math.log10(4*math.pi) + 2*math.log10(dl*3.086) + 48
            mock_blueshift_coat[j] = 299792.458*(mock_centroid-1549.48)/1549.48 #coatman

        sigmaerr[i] = statistics.stdev(second_moment)
        fwhmerr[i] = statistics.stdev(fwhm)
        EW_err[i] = statistics.stdev(mock_EW)
        CIV_lum_err[i] = statistics.stdev(mock_Lciv)
        errblueshift[i] = statistics.stdev(mock_blueshift)
        errblueshift_coat[i] = statistics.stdev(mock_blueshift_coat)

    if flag_StoN:
        StoN[i] = flux1350[i]/noise[i]

if calcola_di_nuovo_tutti_dati:
    with open('uncertainties.txt', 'w+') as fp:
        fp.write("name\tz\tsigma\tsigmaerr\tfwhm\tfwhmerr\tlum1350\tlum1350err+\tlum1350err-\tflux5GHz\tline_lum\ttline_lumerr\tEW\tEWerr\n")
        for i in range(19):
            fp.write("{0}\t{1:.2f}\t{2:.0f}\t{3:.0f}\t{4:.0f}\t{5:.0f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\t{9:.0f}\t{10:.3f}\t{11:.3f}\t{12:.2f}\t{13:.2f}\t{14:.0f}\t{15:.0f}\n".format(names[i],z[i],best_second_moment[i],sigmaerr[i],best_fwhm[i],fwhmerr[i],l1350[i],l1350errh[i],l1350errl[i],radio_flux[i],CIV_lum[i],CIV_lum_err[i],EW[i],EW_err[i], -blueshift[i], errblueshift[i]))

if not calcola_di_nuovo_tutti_dati:
    with open("uncertainties.txt", 'r') as fp:
        lines = fp.readlines()[1:]
        for i, line in enumerate(lines):
        	names.append(line.split()[0])
        	z[i] = float(line.split()[1])
        	best_second_moment[i] = float(line.split()[2])
        	sigmaerr[i] = float(line.split()[3])
        	best_fwhm[i] = float(line.split()[4])
        	fwhmerr[i] = float(line.split()[5])
        	l1350[i] = float(line.split()[6])
        	l1350errh[i] = float(line.split()[7])
        	l1350errl[i] = float(line.split()[8])
        	radio_flux[i] = float(line.split()[9])
        	CIV_lum[i] = float(line.split()[10])
        	CIV_lum_err[i] = float(line.split()[11])
        	EW[i] = float(line.split()[12])
        	EW_err[i] = float(line.split()[13])
    with open('uncertainties.txt', 'w+') as fp:
        fp.write("name\tz\tsigma\tsigmaerr\tfwhm\tfwhmerr\tlum1350\tlum1350err+\tlum1350err-\tflux5GHz\tline_lum\ttline_lumerr\tEW\tEWerr\tblueshift\terrblueshift\tblueshift_Coatman\terr_Coatman\n")
        for i in range(19):
            fp.write("{0}\t{1:.2f}\t{2:.0f}\t{3:.0f}\t{4:.0f}\t{5:.0f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\t{9:.0f}\t{10:.3f}\t{11:.3f}\t{12:.2f}\t{13:.2f}\t{14:.0f}\t{15:.0f}\t{16:.0f}\t{17:.0f}\n".format(names[i],z[i],best_second_moment[i],sigmaerr[i],best_fwhm[i],fwhmerr[i],l1350[i],l1350errh[i],l1350errl[i],radio_flux[i],CIV_lum[i],CIV_lum_err[i],EW[i],EW_err[i], blueshift[i], errblueshift[i], blueshift_coat[i], errblueshift_coat[i]))

if flag_StoN:
    err_relsigma = np.zeros(19)
    err_relfwhm = np.zeros(19)
    with open("StoNratio.txt","w+") as fp:
        fp.write("source\tflux1350\tCIV_noise\tS/N\n")
        for i in range(19):
            fp.write("{0}\t{1:.2f}\t{2:.2f}\t{3:.1f}\n".format(names[i], flux1350[i], noise[i], StoN[i]))
    with open("uncertainties.txt", "r") as fp:
        for i, line in enumerate(fp.readlines()[1:]):
            err_relsigma[i] = (float(line.split()[3])/float(line.split()[2]))*0.5
            err_relfwhm[i] = (float(line.split()[5])/float(line.split()[4]))*0.5
    plt.scatter(StoN, err_relsigma)
    plt.title("S/N vs relative error of sigma")
    plt.xlabel("S/N")
    plt.ylabel(r"$err_{\sigma}/\sigma$")
    plt.savefig("./output/S_N_sigma", format='png', dpi=300)
    plt.clf()
    plt.scatter(StoN, err_relfwhm)
    plt.title("S/N vs relative error of sigma")
    plt.xlabel("S/N")
    plt.ylabel(r"$err_{FWHM}/FWHM$")
    plt.savefig("./output/S_N_fwhm", format='png', dpi=300)
    plt.clf()

if Flag_baldwin:
    plt.scatter(l1350, np.log10(EW))
    p = np.polyfit(l1350, np.log10(EW), 1)
    xcoord = np.arange(45.8, 47.2, 0.1)
    plt.plot(xcoord, p[1]+p[0]*xcoord)
    plt.ylabel(r"$logEW_{CIV}$")
    plt.xlabel(r"$logL_{1350} (erg s^{-1} cm^{-2} \AA^{-1})$")
    plt.title("Baldwin effect")
    plt.savefig("./output/Baldwin_optic", format='png', dip=300)
    plt.clf()
    plt.scatter(radio_flux, np.log10(EW))
    plt.ylabel(r"$logEW_{CIV}$")
    plt.xlabel(r"$logS_{5GHz} (mJy)$")
    plt.xlim((1.35,2.7))
    plt.title("Baldwin effect")
    plt.savefig("./output/Baldwin_radio", format='png', dip=300)
    plt.clf()
