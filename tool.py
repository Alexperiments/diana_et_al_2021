#!/usr/bin/python3
from functions import *

name, z, radio_flux, mag, m_no_CIV, m_CIV = [],[],[],[],[],[]
buffer = []

with open('sdss_in_class.txt') as fp:
	lines = fp.readlines()[1:]
	for line in lines:
		string = line.split()
		if float(string[6]) >= 9:
			m_CIV.append(string[9])

print(len(m_CIV))
########
'''with open('sdss_in_class.txt') as fp:
	lines = fp.readlines()[1:]
	for line in lines:
		buffer.append(line.split())

with open('shen_radio.txt') as fp:
	lines = fp.readlines()[1:]
	for line in lines:
		string = line.split()
		name.append(string[0])
		m_no_CIV.append(string[4])
		m_CIV.append(string[6])

for i in range(len(buffer)):
	for j in range(len(name)):
		if name[j] == buffer[i][0]:
			buffer[i][9] = m_CIV[j]

out_sm = open('sdss_in_class2.txt','w+')
out_sm.write("classname	gbflux	psfmagr	z_temp	z_sdss	logMbh_shen	logMbh_shen_err	logedd_ratio_shen	logMCIV\n")
for i in range(len(buffer)):
	for j in range(len(buffer[i])):
		out_sm.write(buffer[i][j])
		out_sm.write("\t")
	out_sm.write("\n")
out_sm.close()'''

'''def ned_calc(z, H0=70, Omega_m=0.3, Omega_vac=0.7):
    # initialize constants

    WM = Omega_m   # Omega(matter)
    WV = Omega_vac # Omega(vacuum) or lambda
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

    return DL_Mpc, DCMR_Mpc, V_Gpc

print("Digitare il numero corrispondente al tool:\n1) K-Correction calculator\n2) Distance Calculator\n3) Magnitude calculator\nQ - exit")
selector = input("\n")

while (selector == "1") or (selector == "2") or (selector == "3"):

    if selector == "1":
        ################################################## K-Correction ####################################################
        # assuming f(x)=x^-a SED distribution (a spectrale index).
        # Where x is the wavelength. x_o = observed wavelength, x_i = restframe wl f_o = observed flux, f_i = restframe flux
        # Since f_o=x_o^-a => f_i/(1+z) = [x_i*(1+z)]^-a => f_i = x_i^-a * (1+z)^(1-a) => f_i/f_0 = (x_i/x_o)^-a*(1+z)^(1-a)
        # Where (1+z)^(1-a) is the k-correction, so that: f_i(x_i)/f_o(x_o) = K*(x_i/x_o)^-a
        ####################################################################################################################

        alpha = float(input("Spectral index = ") or "0.44")
        obs_wl = float(input("Observed wavelength = ") or "6580")
        rf_wl = float(input("Target wavelength = ") or "1400")
        z = float(input("Redshift = "))
        obs_flux = input("Observed flux = ")

        k_correction = (1+z)**(1-alpha)
        band_conversion = (rf_wl/obs_wl)**(-alpha)

        print("K-Correction = {}".format(k_correction))
        print("band-correction = {}".format(band_conversion))

        if obs_flux != '':
            obs_flux = float(obs_flux)
            rf_flux = obs_flux*band_conversion*k_correction
            print(rf_flux)


        # Se ti dimentichi fai partire questo esempio e sarà tutto più chiaro!

        x_rf = np.arange(1215, 2000, 100)
        f_rf = x_rf**(-alpha)

        x_obs = x_rf*(1+z)
        f_obs = f_rf/(1+z)

        f_band = f_obs*(x_rf/x_obs)**(-alpha)
        f_k = f_band*k_correction

        plt.plot(x_rf,f_rf)
        plt.plot(x_obs,f_obs)
        plt.plot(x_rf,f_band)
        plt.plot(x_rf,f_k)
        plt.show()

    if selector == "2":
        z = float(input("Redshift = "))
        H0 = float(input("Hubble parameter = ") or "70")
        Omega_m = float(input("Omega matter = ") or "0.3")
        Omega_vac = float(input("Omega lambda = ") or "0.7")

        D_Mpc, CD_Mpc, V_Gpc = ned_calc(z,H0,Omega_m,Omega_vac)
        print("Luminosity distance = {} Mpc".format(D_Mpc))
        print("Comoving radial distance = {} Mpc".format(CD_Mpc))
        print("Comoving volume = {} Gpc^3".format(V_Gpc))

    if selector == "3":
        z = float(input("Redshift = "))
        app_magnitude = float(input("Apparent magnitude = ") or "21.5")
        obs_wl = float(input("Observed wavelength = ") or "6580")
        rf_wl = float(input("Target wavelength = ") or "1400")
        alpha = float(input("Spectral index = ") or "0.44")

        k_correction = (1+z)**(1-alpha)
        band_conversion = (rf_wl/obs_wl)**(-alpha)
        DL,_,_ = ned_calc(z)
        logf_nu = -(app_magnitude + 48.60)/2.5
        logf_lambda = logf_nu - 2*math.log10(obs_wl) + math.log10(2.99) + 18
        abs_magnitude = app_magnitude - 5*(math.log10(DL) + 5) + 2.5*math.log10(k_correction*band_conversion)

        print("Specific observed flux nu = {} (Log)".format(logf_nu))
        print("Specific observed flux lambda = {} (Log)".format(logf_lambda))
        print("Absolute magnitude at the target wavelength = {}".format(abs_magnitude))

    selector = input("")
'''
