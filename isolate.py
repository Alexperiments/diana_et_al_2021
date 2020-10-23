#!/usr/bin/python3

from functions import *

########################  SPIKE REMOVAL ROUTINE  ############################

def modified_z_score(intensity):
 median_int = np.median(intensity)
 mad_int = np.median([np.abs(intensity - median_int)])
 modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
 return modified_z_scores

def fixer(y, m, threshold=2.6, flag=False):
    delta_int = np.diff(y)
    intensity_modified_z_score = np.array(np.abs(modified_z_score(delta_int)))
    spikes = abs(np.array(modified_z_score(delta_int))) > threshold
    y_out = y.copy()
    for j in np.arange(len(spikes)-10):
        if spikes[j] != 0: # If we have an spike in position i
            w = np.arange(j-m,j+1+m) # we select 2 m + 1 points around our spike
            w2 = w[spikes[w] == 0] # From such interval, we choose the ones which are not spikes
            y_out[j] = np.mean(y[w2]) # and we average their values
    if flag:
        plt.plot(intensity_modified_z_score, "g--", lw=0.2)
        plt.axhline(y=threshold)
        plt.savefig("./spike_removal/"+names[i]+"_zscore", format="png")
        plt.clf()
    return y_out

def remove_spikes(y, thr):
    ysub = fixer(y, 11, threshold=thr, flag=Flag_spike)
    ysub = fixer(ysub, 9, threshold=thr)
    ysub = fixer(ysub, 7, threshold=thr)
    ysub = fixer(ysub, 5, threshold=thr)
    return fixer(ysub, 3, threshold=thr)

######################  (END)SPIKE REMOVAL ROUTINE  ##########################

EW = np.zeros(19)
l1350 = np.zeros(19)
radio_flux = np.zeros(19)

for i in range(19):
    print("Processing {}...".format(names[i]))
    with open("spettri_txt/"+names[i]+".txt") as fp:
        lines = fp.readlines()[2:]

    wl = np.zeros(len(lines))
    flux = np.zeros(len(lines))
    for l,line in enumerate(lines):
        wl[l] = float(line.split()[0])
        flux[l] = float(line.split()[1])

    x = wl/(1+z[i])
    y = flux*(1+z[i])

    plt.plot(wl, flux, c='black', lw=0.5)
    plt.title(names[i])
    plt.xlabel(r"$\lambda (\AA)$")
    plt.ylabel(r"$f_{\nu} (erg s^{-1} cm^{-2} \AA^{-1})$")
    plt.savefig("./raw_spectra/" + names[i], format="eps")
    plt.clf()

    trim_low = 1350
    trim_high = 1710

    m,q = continuum_estimate(np.vstack((x, y)), bg_int[i])

    y = y[np.where(x > trim_low)[0][0]:np.where(x < trim_high)[0][-1]] # trimming
    x = x[np.where(x > trim_low)[0][0]:np.where(x < trim_high)[0][-1]]
    ycontinuum = m*x+q
    ysub = y - ycontinuum

    masked_y = mask(x, ysub, masks_int[i])
    y_spikeless = remove_spikes(masked_y, threshold[i])

    y_total = savitzky_golay(y_spikeless, 21, 5)
    y_total_sub = y_total + ycontinuum

    if Flag_spike:
        plt.plot(y, 'r--', lw=0.2)
        plt.plot(y_spikeless, c='black', lw=0.5)
        plt.savefig("./spike_removal/" + names[i] + "spike removal", format="png", dpi=300)
        plt.clf()

    if Flag_eps:
        fig, axs = plt.subplots(2, sharex=True,gridspec_kw={'hspace': 0})
        axs[0].plot(x,y,c='darkorange',lw=0.2)
        axs[0].plot(x, y_spikeless + ycontinuum, c='black',lw=0.5)
        for j in range(4):
            axs[0].axvline(x=bg_int[i][j], c='mediumseagreen', ls='--', lw=0.5)
        for j in range(int(len(masks_int[i])/2)):
            axs[0].axvspan(masks_int[i][2*j], masks_int[i][2*j+1], alpha=0.2, color='lightgrey')
        axs[0].plot(x,ycontinuum, c='dodgerblue', ls='-.', lw=0.5)
        plt.xlim((min(x), max(x)+2))

        axs[1].plot(x,y_total,c='black',lw=0.5)
        axs[1].axhline(y=0, c='dodgerblue', ls='-.', lw=0.5)
        ymodel, _, _ = fit_CIV(names[i], x, y_total)
        axs[1].plot(x,ymodel,c='dodgerblue',lw=0.5)

        fig.text(0.5, 0.03, r"Wavelength ($\AA$)", ha='center')
        fig.text(0.04, 0.5, r"Specific flux ($erg s^{-1} cm^{-2} \AA^{-1}$)", va='center', rotation='vertical')
        fig.text(0.5, 0.93, names[i], ha='center')
        plt.savefig("./spectra_preparation/" + names[i] + ".eps", format="eps")
        plt.clf()

    with open("CIV/"+names[i]+".d", "w+") as fp:
        fp.write("# Wavelength\tflux(E-17)\n#\n")
        for i in range(len(x)):
            fp.write("{0}\t{1}\n".format(x[i], y_total_sub[i]))
