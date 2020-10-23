#!/usr/bin/python3
from functions import *
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator

z_bin = np.arange(1,3,0.01)
L_min = np.arange(45, 49.2, 0.1)
maxs = np.zeros(len(L_min))

for i in range(len(L_min)):
    LFs = trap_int_L(L_min[i], L_min[i]+5, z_bin)
    max_value = np.max(LFs)
    maxs[i] = z_bin[np.where(LFs == max_value)]


fig, ax = plt.subplots()
plt.plot(L_min, maxs, color='black')
plt.axvline(M_in_Lbol(9, 1), color='dodgerblue', linewidth=0.75, linestyle='-.', label=r"QLF $\lambda_{edd}=1$")
plt.axvline(M_in_Lbol(9, 0.67), color='dodgerblue', linewidth=0.75, linestyle='-', label=r"QLF $\lambda_{edd}=0.67$")
plt.axvline(M_in_Lbol(9, 0.2), color='dodgerblue', linewidth=0.75, linestyle='--', label=r"QLF $\lambda_{edd}=0.2$")
plt.axhline(2.6, color='red', linewidth=0.75, linestyle='-', label=r"CLASS RL $z_{max}$")
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_locator(  AutoMinorLocator(5))
ax.xaxis.set_minor_locator(  AutoMinorLocator(5))
ax.xaxis.set_tick_params(which='major', direction='inout', length=10)
ax.yaxis.set_tick_params(which='major', direction='inout', length=10)
ax.xaxis.set_tick_params(which='minor', direction='inout', length=5)
ax.yaxis.set_tick_params(which='minor', direction='inout', length=5)
ax.set_xticks(np.arange(round(min(z_bin),2), round(max(z_bin)+(max(z_bin)-min(z_bin))/10, 2), round((max(z_bin)-min(z_bin))/10, 2)))
ax.xaxis.set_major_locator(plt.MaxNLocator(8))
plt.ylim([1,3])
plt.xlim([45,49])
plt.xlabel(r"$logL_{min} (erg s^{-1})$")
plt.ylabel(r"z")
plt.title(r"Redshift of the QLF-peak with $L>logL_{min}$")
plt.grid()
plt.legend()
plt.savefig("./z_qlf_peak.eps" ,dpi=300)
plt.clf()
