import config
import utils
import shen_qlf
import pandas as pd
import numpy as np
import os
import io
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import rcParams
import pickle
import warnings

warnings.filterwarnings("ignore")

def broken_pow_law(z, phi0, z0, alpha, beta):
    return phi0 - np.log10(((1+z)/(1+z0))**alpha + ((1+z)/(1+z0))**beta)

def gehrels_lolim(lower, N, cl):
    return abs(poisson.cdf(N - 1, lower) - cl)

def gehrels_uplim(upper, N, cl):
    return abs(1 - poisson.cdf(N, upper) - cl)

def gehrels(N, cl=0.8413, method='Nelder-Mead'):
    gehrels_lo = minimize(gehrels_lolim, N, args=(N,cl), method=method)['x'][0]
    gehrels_up = minimize(gehrels_uplim, N, args=(N,cl), method=method)['x'][0]
    return [gehrels_lo, gehrels_up]

def stamp_correction_line(z1, z2, s, m, ci, cs):
    s = f"{z1:.2f} - {z2:.2f} & {s:.0f} & {m:.2f} & {ci:.2f} & {cs:.2f} \\\\\n"
    return s

def stamp_density_line(z1, z2, N, n_bl, bl_unc):
    n_5 = np.log10(n_bl*low_g)
    n_10 = np.log10(n_bl*2*gamma**2)
    n_15 = np.log10(n_bl*high_g)
    upp_err = np.log10(1 + bl_unc[1]/n_bl)
    low_err = np.log10(1 + bl_unc[0]/n_bl)
    n_bl = np.log10(n_bl)
    return (f"${z1:.2f} - {z2:.2f}$ & {N:.0f} & {n_bl:.2f}$"
            f"\\substack{{ +{upp_err:.2f} \\\\ {low_err:.2f} }}$ & {n_5:.2f}"
            f" & {n_10:.2f} & {n_15:.2f} \\\\\n")

def stamp_fit_parameters(phi, z, g1, g2, phi_e, z_e, g1_e, g2_e):
    phi = phi + np.log10(2*gamma**2)
    print(
        f"Log$n_0={phi:.2f}\\substack{{ +{phi_e[1]:.2f} \\\\ {phi_e[0]:.2f}}}$"
        f" $z_0={z:.2f}\\substack{{ +{z_e[1]:.2f} \\\\ {z_e[0]:.2f}}}$ "
        f"$\gamma_1={g1:.2f}\\substack{{ +{g1_e[1]:.2f} \\\\ {g1_e[0]:.2f}}}$"
        f" $\gamma_2={g2:.2f}\\substack{{ +{g2_e[1]:.2f} \\\\ {g2_e[0]:.2f}}}$"
    )

# bins limits
z_bin = [1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5]
#normalizzazione dell'area SDSS/CLASS z<4 e scaling del 94% per il misclass.
norm_near = 41253*0.94/10700
#normalizzazione dell'area SDSS/CLASS z>4
norm_far = 41253/13120
# preferred lorentz bulk factor
gamma = 10
# conversion factor for the lower lorentz bulk factor
low_g = 2*5**2
# conversion factor for the upper lorentz bulk factor
high_g = 2*15**2
# number of Monte Carlo extractions during the uncertainties estimate
Ntry = 100000

# build a vector containing the normalizations coeff
norm = np.concatenate(
    (
        np.full(shape=(1, 8), fill_value=norm_near),
        np.full(shape=(1, 3), fill_value=norm_far),
    ),
    axis=None
)
# build a vector containing the widths of the bins
widths = np.concatenate(
    (
        np.full(shape=(1, 6), fill_value=0.25),
        np.full(shape=(1, 5), fill_value=0.5),
    ),
    axis=None
)

# Read the physical information and the estimated parameters of the objects
with open(config.PARAMETERS_FILE, 'rb') as f:
    parameters = pickle.load(f)
class_data = pd.read_csv(config.SDSS_IN_CLASS_FILE, sep='\t')
full_sample = pd.read_csv(config.SELECTION_FILE, sep='\t')
full_sample_z = full_sample['z'].values
with_spectrum = parameters['z'][ parameters['Msigma'] > 0].values

# Print useful info about the sample numbers
print("####### How many objects... #######")
print(f"in the full sample: {len(full_sample)}")
print(f"with z<4: {len(full_sample_z[full_sample_z<4])}")
print(f"with spectrum: {len(parameters['Msigma'])}")
print(f"with M>10^9Msun: {len(parameters['Msigma'][parameters['Msigma'] >= 9])}")

with_id = class_data[ (class_data['flag_class'] != 0) ]
without_id = class_data[ (class_data['flag_class'] == 0) ]

# initialize the needed arrays
n_bin = len(z_bin)-1
blaz_dens       = np.zeros(n_bin)
RL_dens         = np.zeros(n_bin)
mean_z          = np.zeros(n_bin)
percent_unc     = np.zeros((n_bin, 2))
RL_unc          = np.zeros((n_bin, 2))
blaz_unc        = np.zeros((n_bin, 2))
MC_densities    = np.zeros((n_bin, Ntry))

# create the buffers to store the tables
corr_table_buffer = io.StringIO()
dens_table_buffer = io.StringIO()

# iterate over the bins to calculate corrections, counts, densities and errors
for i in range(n_bin):
    z1 = z_bin[i]
    z2 = z_bin[i+1]
    mean_z[i] = (z2 + z1)*0.5
    mean_radio_limit, mean_mag_limit = utils.rad_opt_limits(mean_z[i])

    # number of objs with id in the bin
    yes_ID = sum(
        (with_id['psfmagr'] <= mean_mag_limit)  &
        (with_id['gbflux'] >= mean_radio_limit)
    )
    # number of objs with id in the bin
    no_ID = sum(
        (without_id['psfmagr'] <= mean_mag_limit) &
        (without_id['gbflux'] >= mean_radio_limit)
    )
    if z1 < 4: C_ID = (no_ID+yes_ID)/yes_ID
    elif z1 < 4.5: C_ID = 1#/0.92 # The C19 sample is complete up to 92% at 4<z<4.5
    else: C_ID = 1

    # number of selected objects in the bin
    obj_inbin = sum((full_sample_z >= z1) & (full_sample_z < z2))
    # number of selected objects in the bin with spect
    with_spectrum_inbin = sum((with_spectrum >= z1) & (with_spectrum < z2))
    if z1 < 4: C_spect = obj_inbin/with_spectrum_inbin
    else: C_spect = 1

    # select the masses of the objects in the bin
    z_mask = (
        (parameters['z'] >= z1)   &
        (parameters['z'] < z2)
    )
    masses_inbin = parameters['Msigma'][z_mask]

    # calculate the number of obj with M>=10^9Msun
    count_inbin = sum(masses_inbin >= 9)
    # correct for the missing id and spectra
    corrected_count = count_inbin*C_ID*C_spect
    corr_table_buffer.write(
        stamp_correction_line(
            z1, z2, mean_radio_limit, mean_mag_limit, C_ID, C_spect
        )
    )

    # Monte Carlo simulation
    MC_masses = np.random.normal(masses_inbin, 0.3, size=(Ntry, len(masses_inbin)))
    MC_select = MC_masses >= 9
    MC_count = np.random.poisson(np.sum(MC_select, axis=1))
    MC_CID = (no_ID+yes_ID)/np.random.binomial(no_ID+yes_ID, 1/C_ID, size=Ntry)
    MC_Cspect = obj_inbin/np.random.binomial(obj_inbin, 1/C_spect, size=Ntry)
    MC_corr = MC_CID*MC_Cspect

    _, v2 = utils.ned_calc(z2)
    _, v1 = utils.ned_calc(z1)
    volumes = v2 - v1

    MC_densities[i] = MC_count*MC_corr*norm[i]/volumes
    blaz_dens[i] = corrected_count*norm[i]/volumes
    RL_dens[i] = blaz_dens[i]*2*gamma*gamma

    if z1 < 4.5:
        unc_on_counts = np.quantile(MC_count*MC_corr, q=(0.16, 0.84))
        percent_unc[i,:] = unc_on_counts/np.quantile(MC_count*MC_corr, q=0.5)#corrected_count
    else: percent_unc[i,:] = gehrels(count_inbin)/np.array(count_inbin)

    RL_unc[i,:] = RL_dens[i]*(percent_unc[i,:] - 1)
    blaz_unc[i,:] = blaz_dens[i]*(percent_unc[i,:] - 1)
    dens_table_buffer.write(
        stamp_density_line(
            z1, z2, count_inbin, blaz_dens[i], blaz_unc[i,:]
        )
    )

z_QLF = np.arange(1, 6.6, 0.1)
min_lum_eff = shen_qlf.calc_lum_eff()
QLF_mean = 10**shen_qlf.shen_qlf(z_QLF, min_lum_eff, min_lum_eff + 4)

MC_RLF = []
MC_pars = []
max_z = []
z_RLF = np.arange(1.5, 6.6, 0.1)

for i in range(5000):
    mock_par, _ = curve_fit(
        broken_pow_law, mean_z, np.log10(MC_densities[:,i]), maxfev = 10000, check_finite=False,
        #sigma= np.mean(RL_unc, axis=1),
    )
    # swap back the two gammas if they happen to be inverted
    if mock_par[2] < 2:
        temp = mock_par[3]
        mock_par[3] = mock_par[2]
        mock_par[2] = temp

    # save the parameters, the RLF and the z_peak for the N extracted samples
    rlf = np.array(broken_pow_law(z_RLF, *mock_par))
    max_rlf = max(rlf)

    # save only the fits that successfully captured the shape of the points
    if z_RLF[rlf==max_rlf] > 1.5:
        MC_RLF.append(rlf)
        MC_pars.append(mock_par)
        max_z.append(*z_RLF[rlf==max_rlf])

param_median = np.quantile(MC_pars, q=0.5, axis=0)
param_1sigma = np.quantile(MC_pars, q=(0.16, 0.84), axis=0) - param_median
peak_z = np.mean(max_z)
peak_z_1sigma = np.quantile(max_z, q=(0.16, 0.84), axis=0) - np.mean(max_z)

print("####### Corrections table #######")
print(corr_table_buffer.getvalue())
print("####### Densities table #######")
print(dens_table_buffer.getvalue())
print('####### Fit parameters #######')
stamp_fit_parameters(*param_median, *param_1sigma.T)
print('\n####### RL z peak #######')
print(f"${peak_z:.2f} \\substack{{ {peak_z_1sigma[1]:.2f} \\\\ {peak_z_1sigma[0]:.2f}}}$")

RLF_high = np.mean(MC_RLF, axis=0) + np.std(MC_RLF, axis=0)
RLF_low = np.mean(MC_RLF, axis=0) - np.std(MC_RLF, axis=0)
RLF_mean = np.mean(MC_RLF, axis=0)

config.default_plot_settings()

fig, ax = plt.subplots(1, 1)
ax.set_box_aspect(1)
'''ax.errorbar(
    mean_z, RL_dens, xerr=widths*0.5, yerr=abs(RL_unc.T), fmt='D', color='k',
    elinewidth=0.7, markersize=3, capsize=2, capthick=0.7, zorder=10
)
ax.errorbar(
    6, 1.10, xerr=0.5, yerr=[[0.91], [2.53]], fmt='D', color='magenta',
    elinewidth=0.7, markersize=3, capsize=2, capthick=0.7
)'''
ax.arrow(2, 10**0.5, 0, 10**1.9,
         overhang = -0.1, head_width=0.1, head_length=20,
         width = 0.05, alpha=0.75)
ax.text(
    2.5, 10**1.2, r"$\times2\Gamma^2$", ha="center", va="center", size=15)
ax.errorbar(
    mean_z, blaz_dens, xerr=widths*0.5, yerr=abs(blaz_unc.T), fmt='D', color='k',
    elinewidth=0.7, markersize=3, capsize=2, capthick=0.7
)
ax.errorbar(
    6, 1.10/200, xerr=0.5, yerr=[[0.91/(2*gamma*gamma)], [2.53/(2*gamma*gamma)]], fmt='D', color='magenta',
    elinewidth=0.7, markersize=3, capsize=2, capthick=0.7
)
ax.fill_between(z_RLF, 10**RLF_low*2*gamma**2, 10**RLF_high*2*gamma**2, color='#5681b9', label=r'$\Gamma=10$', alpha=0.5)
ax.plot(z_QLF, QLF_mean, color="#93003A")
plt.yscale('log')
ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
ax.set_yticklabels(['-3', '-2', '-1', '0', '1', '2', '3'])
minor_ticks = np.logspace(-3, 3, num=31)
ax.set_yticks(minor_ticks, minor=True)
plt.ylim([0.001, 1000])
plt.xlim([1, 6.5])
plt.xlabel('z')
plt.ylabel(r"Log$n_{jet}$ (M$_{BH}$>10$^9$M$_{\odot}$) [cGpc$^{-3}$]")
leg = ax.legend(loc='upper right', fontsize='medium', frameon='False', framealpha=0)
leg.get_frame().set_linewidth(0.0)
plt.savefig(config.PAPER_PLOTS + "density.pdf")
plt.close(fig)

QLF_mean_tofraction = 10**shen_qlf.shen_qlf(z_RLF, min_lum_eff, min_lum_eff + 4)
low_fraction = 10**RLF_low/QLF_mean_tofraction
high_fraction = 10**RLF_high/QLF_mean_tofraction
mean_fraction = 10**RLF_mean/QLF_mean_tofraction

print(mean_fraction[0]/mean_fraction[-1])

fig, ax = plt.subplots(1, 1)
ax.set_box_aspect(1)
ax.fill_between(z_RLF, low_fraction*high_g, high_fraction*high_g, color='#93c4d2', alpha=0.5, label=r"$\Gamma=15$")
ax.fill_between(z_RLF, low_fraction*2*gamma**2, high_fraction*2*gamma**2, color='#5681b9', alpha=0.5, label=r"$\Gamma=10$")
ax.fill_between(z_RLF, low_fraction*low_g, high_fraction*low_g, color='#002A7F', alpha=0.5, label=r"$\Gamma=5$")
plt.yscale('log')
ax.errorbar([6], [0.119], xerr=[[0.25],[0.25]], yerr=[[0.053],[0.053]], fmt='D', color='orange', markersize=6, elinewidth=0.9, capsize=2, capthick=0.9, label="Liu (2021)")  # Liu 2021
ax.errorbar(5, 0.071, xerr=[[0.3],[0.4]], yerr=[[0],[0.05]], fmt='x', color='red', markersize=8, lolims=True, elinewidth=0.9, capsize=2, capthick=0.9, label="Yang (2016)")  # Yang 2016
ax.errorbar([5.95], [0.081], xerr=[[0.45],[0.45]], yerr=[[0.032],[0.05]], fmt='*', color='blue', markersize=8, elinewidth=0.9, capsize=2, capthick=0.9, label="Bañados (2015)")  # Banados 2015
ax.plot([1.53 for _ in range(11)], np.arange(0.10, 0.21, 0.01), alpha=0.75, lw=8, solid_capstyle='round', color='forestgreen', zorder=1, label='local')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
plt.yticks(fontsize=12)
plt.xlabel('z')
plt.ylabel(r"Jetted AGNs fraction (M$_{BH}$>10$^9$M$_{\odot}$)")
plt.ylim([0.01, 1])
plt.xlim([1.5, 6.5])
leg = ax.legend(loc='lower left', fontsize='x-small', frameon='False', framealpha=0, ncol=2, borderpad=0.1, labelspacing=0.2, columnspacing=0.2)
leg.get_frame().set_linewidth(0.0)
plt.savefig(config.PAPER_PLOTS + "fraction.pdf")
