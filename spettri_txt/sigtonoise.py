#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import statistics
import math

import pylab as P
import scipy

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def s2n_ratio(x, y, min, max):
    signal_window = [s for i,s in zip(x,y) if (i > min and i < max)]
    signal = statistics.mean(signal_window)
    noise = statistics.stdev(signal_window)
    snr = signal/noise
    return snr

names = ['GB6J001115+144608','GB6J003126+150729','GB6J012126+034646','GB6J083548+182519',
'GB6J083945+511206','GB6J091825+063722','GB6J102107+220904','GB6J102623+254255','GB6J132512+112338',
'GB6J134811+193520','GB6J141212+062408','GB6J143023+420450','GB6J151002+570256','GB6J153533+025419',
'GB6J161216+470311','GB6J162956+095959','GB6J164856+460341','GB6J171103+383016','GB6J235758+140205']

z =  [4.94,4.29,4.13,4.41,4.4,4.17,4.26,5.27,4.40,4.38,4.37,4.72,4.31,4.37,4.33,
4.97,5.36,3.98,4.25]

snrlow = 1430
snrhigh = 1460
sigma = np.zeros(19)
fwhm = np.zeros(19)
errsigma = np.zeros(19)
errfwhm = np.zeros(19)
snr = np.zeros(19)
wl = []
fl = []

with open("masse_python.txt", 'r') as fp:
    for i,line in enumerate(fp.readlines()[1:]):
        sigma[i] = float(line.split()[2])
        fwhm[i] = float(line.split()[4])
        errsigma[i] = float(line.split()[3])
        errfwhm[i] = float(line.split()[5])

for i,file in enumerate(names):
    x = []
    y = []
    with open(file + ".txt") as fp:
        for line in fp.readlines()[2:]:
            x.append(float(line.split()[0]))
            y.append(float(line.split()[1]))
        wl.append(np.asarray(x)/(1+z[i]))
        fl.append(np.asarray(y)*(1+z[i]))
        snr[i] = s2n_ratio(wl[i],fl[i],snrlow,snrhigh)

intervals = 150
S = np.zeros([intervals,19])
sigtonoise = np.zeros([intervals,19])

for k in range(intervals):
    snrlow = 1350 + k
    snrhigh = 1370 + k
    print(k)
    for i in range(19):
        sigtonoise[k][i] = s2n_ratio(wl[i],fl[i],snrlow,snrhigh)
    m,b = np.polyfit(sigtonoise[k][:], errfwhm/fwhm, 1)
    for j in range(19):
        S[k][j] = (errfwhm[j]/fwhm[j] - sigtonoise[k][j]*m - b)**2

y = np.zeros(19)
positions = sigtonoise.argmax(axis=0)[:]
for i in range(19):
    y[i] = sigtonoise[sigtonoise.argmax(axis=0)[i]][i]

for i,file in enumerate(names):
    plt.plot(wl[i],fl[i],c='grey',lw=0.3)
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.ylabel(r"Specific flux ($erg s^{-1} cm^{-2} \AA^{-1}$)")
    plt.title(file + " S/N " + "{0:.1f}".format(y[i]))
    plt.axvline(1350 + positions[i], c='grey', ls='--', lw=0.2)
    plt.axvline(1370 + positions[i], c='grey', ls='--', lw=0.2)
    plt.savefig(file, format="eps")
    plt.cla()

plt.scatter(y, errfwhm/fwhm)
plt.xlabel("S/N")
plt.ylabel("err rel. FWHM")
plt.savefig("s2n.png")
