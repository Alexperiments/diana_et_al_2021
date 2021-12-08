import numpy as np
import matplotlib.pyplot as plt
import os
import config
import pickle

# Fact: l'indice spettrale non dipende dal sistema di riferimento.
# Fact: se αν è l'indice ottico dello spettro in funzione della frequenza, allora αλ=-αν-2
# Fact: l'indice spettrale dello spettro invariante è sempre +1 rispetto a quello normale, quindi per λLλ è αλ+1, mentre per νLν è αν+1.

c = 299792458 # m/s
c_A = c*1e10

alpha_nu = -config.ALPHA_O
alpha_lam = -alpha_nu - 2

obj_name = 'GB6J001837+041532'
file_path = os.path.join(config.OUT_FOLDER, obj_name)
with open(file_path+ '/spectrum.pkl', 'rb') as f:
    obj_spectra = pickle.load(f)
with open(file_path+ '/info.pkl', 'rb') as f:
    obj_info = pickle.load(f)

z = obj_info['z'].values
flux_lam = obj_spectra['flux']
lam = obj_spectra['lambda']

flux_nu = flux_lam*lam**2/c_A
nu = c_A/lam

lamflux_lam = lam*flux_lam
nuflux_nu = nu*flux_nu

alpha_inv_lam = alpha_lam + 1
lamSlam = 8e5*np.power(lam, alpha_inv_lam)

alpha_inv_nu = alpha_nu + 1
nuSnu = 3.5e-5*np.power(nu, alpha_inv_nu)

print(alpha_inv_lam, alpha_inv_nu)

plt.plot(lam, lamflux_lam)
plt.plot(lam, lamSlam)
plt.show()

plt.plot(nu, lamflux_lam)
plt.plot(nu, nuSnu)
plt.show()
