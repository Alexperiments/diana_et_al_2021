#!/usr/bin/python3

from functions import *
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator

fit_A()

def log_error_bar(centr_value, lowe, high):
    upper = high
    lower = lowe
    centr_value = centr_value
    return -np.log10(lower/centr_value), np.log10(upper/centr_value)
def ned_calc(z, H0=70, Omega_m=0.3, Omega_vac=0.7):
	# initialize constants

	WM = Omega_m   # Omega(matter)
	WV = Omega_vac # Omega(vacuum) or lamd
	WR = 0.        # Omega(radiation)
	WK = 0.        # Omega curvaturve = 1-Omega(total)
	c = 299792.458 # velocity of light in km/sec
	Tyr = 977.8    # coefficent for converting 1/H into Gyr
	DTT = 0.5      # time from z to now in units of 1/H0
	DTT_Gyr = 0.0  # value of DTT in Gyr
	age = 0.5      # age of Universe in units of 1/H0
	age_Gyr = 0.0  # value of age in Gyr
	zage = 0.1     # age of Universe at redshift z in units of 1/H0
	zage_Gyr = 0.0 # value of zage in Gyr
	DCMR = 0.0     # comoving radial distance in units of c/H0
	DCMR_Mpc = 0.0
	DCMR_Gyr = 0.0
	DA = 0.0       # angular size distance
	DA_Mpc = 0.0
	DA_Gyr = 0.0
	kpc_DA = 0.0
	DL = 0.0       # luminosity distance
	DL_Mpc = 0.0
	DL_Gyr = 0.0   # DL in units of billions of light years
	V_Gpc = 0.0
	a = 1.0        # 1/(1+z), the scale factor of the Universe
	az = 0.5       # 1/(1+z(object))

	h = H0/100.
	WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
	WK = 1-WM-WR-WV
	az = 1.0/(1+1.0*z)
	age = 0.
	n=1000         # number of points in integrals
	for i in range(n):
		a = az*(i+0.5)/n
		adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
		age = age + 1./adot

	zage = az*age/n
	zage_Gyr = (Tyr/H0)*zage
	DTT = 0.0
	DCMR = 0.0

	# do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
	for i in range(n):
		a = az+(1-az)*(i+0.5)/n
		adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
		DTT = DTT + 1./adot
		DCMR = DCMR + 1./(a*adot)

	DTT = (1.-az)*DTT/n
	DCMR = (1.-az)*DCMR/n
	age = DTT+zage
	age_Gyr = age*(Tyr/H0)
	DTT_Gyr = (Tyr/H0)*DTT
	DCMR_Gyr = (Tyr/H0)*DCMR
	DCMR_Mpc = (c/H0)*DCMR

	# tangential comoving distance

	ratio = 1.00
	x = np.sqrt(abs(WK))*DCMR
	if x > 0.1:
		if WK > 0:
		  ratio =  0.5*(exp(x)-exp(-x))/x
		else:
		  ratio = sin(x)/x
	else:
		y = x*x
		if WK < 0: y = -y
		ratio = 1. + y/6. + y*y/120.
	DCMT = ratio*DCMR
	DA = az*DCMT
	DA_Mpc = (c/H0)*DA
	kpc_DA = DA_Mpc/206.264806
	DA_Gyr = (Tyr/H0)*DA
	DL = DA/(az*az)
	DL_Mpc = (c/H0)*DL
	DL_Gyr = (Tyr/H0)*DL

	# comoving volume computation

	ratio = 1.00
	x = np.sqrt(abs(WK))*DCMR
	if x > 0.1:
		if WK > 0:
		  ratio = (0.125*(exp(2.*x)-exp(-2.*x))-x/2.)/(x*x*x/3.)
		else:
		  ratio = (x/2. - sin(2.*x)/4.)/(x*x*x/3.)
	else:
		y = x*x
		if WK < 0: y = -y
		ratio = 1. + y/5. + (2./105.)*y*y
	VCM = ratio*DCMR*DCMR*DCMR/3.
	V_Gpc = 4.*math.pi*((0.001*c/H0)**3)*VCM

	return V_Gpc
def up_frac(n1, n2, S):
    CL=0.84
    p1u = 0
    if n2>0:
        lamd = (S*S -3)/6
        h=( (1/(2*n2-1)) + (1/(2*n1+1)))
        h=2/h
        w=(S*(h+lamd)**0.5)/h
        w=w+( (1/(2*n2-1) - 1/(2*n1+1)) * (lamd+(5/6) - (2/(3*h))) )
        eps=0.64*(1-S)*math.exp(-1*n2)
        p1u = ((n1+1)*math.exp(2*w)+eps*n2)/((n1+1)*math.exp(2*w)+n2)

        if (n2==0) & (n1>0):
            p1u = 1

        if (n1==0) & (n2>0):
            p1u = 1 - (1-CL)**(1/n2)

        if (n2==1):
            p1u = CL**(1/(n1+n2))
    return p1u
def poisson_68(Num):
    S=1.0
    CL=0.84
    norma=100000
    n1=Num
    n2=norma-n1
    upper_n1=n1-n1
    lower_n1=upper_n1
    p1u = up_frac(n1, n2, S)
    upper_n1= p1u
    p1u = up_frac(n2, n1, S)
    lower_n1=1-p1u
    frac=n1/(n1+n2)
    low=lower_n1*norma
    high=upper_n1*norma
    return low, high

z_low = []
dens_low = []
rang=[]
normalization = 13120/41253
with open("./conteggi.txt", "r") as fp:
    lines = fp.readlines()
    for i,line in enumerate(lines[:-1]):
        rang.append((float(lines[i+1].split()[0])) - (float(lines[i].split()[0])))
        z_low.append((float(lines[i+1].split()[0]) + float(lines[i].split()[0]))/2)
    for line in lines[:-1]:
        dens_low.append(float(line.split()[1]))

vol_low = np.zeros(len(z_low))
errs = np.zeros([len(z_low),2])
logerrs = np.zeros([len(z_low),2])

for i in range(len(dens_low)):
    bin_range = rang[i]*0.5
    low = ned_calc(z_low[i] - bin_range)
    high = ned_calc(z_low[i] + bin_range)
    vol_low[i] = high-low
    errs[i] = poisson_68(dens_low[i])
    logerrs[i] = log_error_bar(dens_low[i], errs[i][0], errs[i][1])
    dens_low[i] = np.log10(dens_low[i]*200/(vol_low[i]*normalization))

z_bin = np.arange(0,7,0.1)
density_up = trap_int_L(M_in_Lbol(9, 0.2), M_in_Lbol(13, 0.2), z_bin)
density_mean = trap_int_L(M_in_Lbol(9, 0.68), M_in_Lbol(13, 0.68), z_bin)
density_down = trap_int_L(M_in_Lbol(9, 1), M_in_Lbol(13, 1), z_bin)
density_try = trap_int_L(46.40, 50.4, z_bin)

err = poisson_68(10)
err_45 = log_error_bar(10, err[0], err[1])
err = poisson_68(1)
err_55 =log_error_bar(1, err[0], err[1])

fig, ax = plt.subplots()
plt.plot(z_bin, density_up, c='dodgerblue', ls='--', lw=0.75, label=r"$\lambda_{Edd}=0.2$")
plt.plot(z_bin, density_down, c='dodgerblue', ls='-.', lw=0.75, label=r"$\lambda_{Edd}=1$")
plt.plot(z_bin, density_mean, c='blue', label=r"$\bar{\lambda}_{Edd}\simeq0.67$")
plt.fill_between(z_bin,density_up, density_down, label='RQ+RL spatial density', alpha=0.05, color='lightgrey')
for i in range(len(dens_low)):
    bin_range = 0.5*rang[i]
    plt.errorbar(z_low[i], dens_low[i], xerr=[[bin_range],[bin_range]], yerr=[[logerrs[i][0]],[logerrs[i][1]]], fmt='ro', markersize=5, elinewidth=1, capsize=3)
plt.errorbar(x=[4.5, 5.5], y=[np.log10(16.4), np.log(1.64)], xerr=[[0.5,0.5],[0.5,0.5]], yerr=[[err_45[0],err_55[0]],[err_45[1],err_55[1]]], fmt='ko', markersize=5, elinewidth=1, capsize=3, label='RL spatial density (CLASS)')
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_locator(  AutoMinorLocator(5))
ax.xaxis.set_minor_locator(  AutoMinorLocator(5))
ax.xaxis.set_tick_params(which='major', direction='inout', length=10)
ax.yaxis.set_tick_params(which='major', direction='inout', length=10)
ax.xaxis.set_tick_params(which='minor', direction='inout', length=5)
ax.yaxis.set_tick_params(which='minor', direction='inout', length=5)
ax.set_xticks(np.arange(round(min(z_bin),2), round(max(z_bin)+(max(z_bin)-min(z_bin))/10, 2), round((max(z_bin)-min(z_bin))/10, 2)))
ax.xaxis.set_major_locator(plt.MaxNLocator(8))
plt.xlim([0,7])
plt.ylim([-2,4])
plt.xlabel("z")
plt.ylabel(r"$log\phi (M>10^9 M_{\odot}) [Gpc^{-3}]$")
plt.legend()
plt.savefig("./density.eps" ,dpi=300)
plt.clf()

'''plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False

z_bin = np.arange(0, 7, 0.01)
plt.subplot(221)
plt.tight_layout()
plt.plot(z_bin, gamma_1(z_bin), c='purple', label="Fit A")
fit_B()
plt.plot(z_bin, gamma_1_Hopkins(z_bin), c='pink', label="Fit B")
plt.xlabel("z")
plt.ylabel(r"$\gamma_1$")
plt.xlim([0,7])
plt.ylim([0.0,1.5])
plt.legend()
fit_A()

plt.subplot(222)
plt.tight_layout()
plt.plot(z_bin, gamma_2(z_bin), c='purple', label="Fit A")
fit_B()
plt.plot(z_bin, gamma_2(z_bin), c='pink', label="Fit B")
plt.xlabel("z")
plt.ylabel(r"$\gamma_2$")
plt.xlim([0,7])
plt.ylim([1.0,3.0])
plt.legend()
fit_A()

plt.subplot(223)
plt.tight_layout()
plt.plot(z_bin, logL_star(z_bin), c='purple', label="Fit A")
fit_B()
plt.plot(z_bin, logL_star(z_bin), c='pink', label="Fit B")
plt.xlabel("z")
plt.ylabel(r"$logL_{*}$")
plt.xlim([0,7])
plt.ylim([10.5,13.5])
plt.legend()
fit_A()

plt.subplot(224)
plt.tight_layout()
plt.plot(z_bin, logPhi_star(1 + z_bin), c='purple', label="Fit A")
fit_B()
plt.plot(z_bin, logPhi_star(1 + z_bin), c='pink', label="Fit B")
plt.xlabel("z")
plt.ylabel(r"$log \Phi_{*}$")
plt.xlim([0,7])
plt.ylim([-6.5,-3.5])
plt.legend()
fit_A()

plt.savefig("LF_param.png", dpi=300)
plt.clf()'''
