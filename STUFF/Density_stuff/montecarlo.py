#!/usr/bin/python3
'''import numpy as np
import matplotlib.pyplot as plt
import statistics 
import math

from astropy.modeling import models
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.integrate import quad

larghezza = 14.13
errore_larghezza = 3.62

velocita = larghezza*299792/1548
errvelocita = errore_larghezza*299792/1548

luminosita = 153.7   
errluminosita = 17.59

Ntries = 1000
masse = np.zeros(Ntries)
for i in range(Ntries):
	v = np.random.normal(velocita, errvelocita)
	l = np.random.normal(luminosita, errluminosita)

	masse[i] = 6.73 + 2*math.log10(v/1000) + 0.53*math.log10(l)
	
print(statistics.stdev(masse))
plt.hist(masse, bins = 20)
plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

larghezza = 14.13
errore_larghezza = 3.62

velocita = larghezza*299792/1548
errvelocita = errore_larghezza*299792/1548

luminosita = 153.7   
errluminosita = 17.59
Ntries = 10000
masse = np.zeros(Ntries)

def update(val):
		sig = ssigma.val
		for i in range(Ntries):
			v = np.random.normal(velocita, sig)
			l = np.random.normal(luminosita, errluminosita)
			masse[i] = 10**6.73*2*(v/1000)**2*(l)**0.53

		ax.cla()
		ax.hist(masse, bins=20)
		ax.autoscale(axis='x')
		plt.xlabel(r"$M (M_{\odot})$")
		plt.draw()


def reset(event):
    mv.reset()
    stdv.reset()
    n_sample.reset()


ax = plt.subplot(111)
plt.subplots_adjust(bottom=0.25)

for i in range(Ntries):
	v = np.random.normal(velocita, errvelocita)
	l = np.random.normal(luminosita, errluminosita)
	masse[i] = 10**6.73*2*(v/1000)**2*(l)**0.53
	
plt.hist(masse, bins=20)
plt.xlabel(r"$M (M_{\odot})$")
axsgima = plt.axes([0.25, 0.10, 0.65, 0.03])

ssigma = Slider(axsgima, 'Err Sigma', 0, 1300, valinit=1281)
ssigma.on_changed(update)

plt.show()
