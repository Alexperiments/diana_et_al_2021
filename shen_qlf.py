import matplotlib.pyplot as plt
import scipy.integrate as integrate
import config
import pickle
import numpy as np

def calc_lum_eff():
    with open(config.PARAMETERS_FILE, 'rb') as f:
        parameters = pickle.load(f)
    N_massive = len(parameters[parameters['Msigma']>=9])
    Lbol = np.sort(parameters['Lbol'][parameters['Lbol']>0])
    return Lbol[-N_massive]

if config.SHEN_FIT == 'A':
    #Global fit A
    a0, a1, a2 = [0.8569, -0.2614, 0.0200]
    b0, b1, b2 = [2.5375, -1.0425, 1.1201]
    c0, c1, c2 = [13.0088, -0.5759, 0.4554]
    d0, d1 = [-3.5426, -0.3936]
    x0 = 2

if config.SHEN_FIT == 'B':
    #Global fit B
    a0, a1 = [0.3653, -0.6006]
    b0, b1, b2 = [2.4709, -0.9963, 1.0716]
    c0, c1, c2 = [12.9656, -0.5758, 0.4698]
    d0, d1 = [-3.6276, -0.3444]
    x0 = 2

#Chebyshev polynomials 0,1,2-th orders
def T_0(x):
    return 1
def T_1(x):
    return x
def T_2(x):
    return 2*x*x-1

def gamma_1(x):
    if config.SHEN_FIT == 'A':
        return a0*T_0(1+x) + a1*T_1(1+x) + a2*T_2(1+x)
    if config.SHEN_FIT == 'B':
        return a0*((1+x)/(1+x0))**a1
def gamma_2(x):
    return 2*b0/( ((1+x)/(1+x0))**b1 + ((1+x)/(1+x0))**b2 )
def logL_star(x):
    return 2*c0/( ((1+x)/(1+x0))**c1 + ((1+x)/(1+x0))**c2 )
def logPhi_star(x):
    return d0*T_0(x) + d1*T_1(x)
def L_star(x):
    return 10**(2*c0/( ((1+x)/(1+x0))**c1 + ((1+x)/(1+x0))**c2 ))
def Phi_star(x):
    return 10**(d0*T_0(x) + d1*T_1(x))

def Phi_bol(z, L):
	return Phi_star(1 + z)/( 10**(gamma_1(z)*(L - logL_star(z))) +10**(gamma_2(z)*(L - logL_star(z))))

def shen_qlf(z, Lminim, Lmaxim):
    Lmin = Lminim -np.log10(3.9) - 33
    Lmax = Lmaxim -np.log10(3.9) - 33
    L_bin = np.linspace(Lmin, Lmax, int(1000*(Lmaxim-Lminim)))
    result = np.zeros(np.size(z))
    dlogL = L_bin[1] - L_bin[0]
    if np.size(z) == 1:
        result = integrate.trapz(Phi_bol(z,L_bin), dx=dlogL)
    else:
        for i,z_bin in enumerate(z):
            result[i] = integrate.trapz(Phi_bol(z_bin,L_bin), dx=dlogL)
    return np.log10(result) + 9

def M_in_Lbol(M, Redd):
    return M + np.log10(1.26) + 38 + Redd


if __name__ == '__main__':
    z_inf, z_sup = 0,7
    z_QLF = np.arange(z_inf, z_sup, 0.1)
    min_lum_eff = calc_lum_eff()
    QLF_mean = shen_qlf(z_QLF, min_lum_eff, min_lum_eff + 4)
    plt.plot(z_QLF, QLF_mean)
    plt.show()
