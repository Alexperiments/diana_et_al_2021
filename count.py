#!/usr/bin/python3
from functions import *

def write_counts(x_redshift, count):
	'''Crea un file 'conteggio_per_sm.txt' con i seguenti dati:
			z medio bin   #   Oggetti con massa > 10^9   #   Larghezza del bin (una riga)
	'''
	out_sm = open('conteggi.txt','w+')
	for i in range(len(count)):
		out_sm.write("{0}\t{1:.2f}\n".format(x_redshift[i], count[i]))
	out_sm.write(str(x_redshift[-1]))
	out_sm.close()
def sdss_compl_calc(z_min, z_max):
	''' Calcola la frazione di oggetti non usati da Shen ma comunque presenti nella SDSS in un determinato bin'''
	found = 0
	total = 0
	with open('sdss_in_class.txt') as sdss:
		next(sdss)
		mean_z = (z_min + z_max)*0.5
		flux_limit, mag_limit = flux_mag_limits(mean_z)
		for line in sdss:
			string = line.split()
			z_SDSS = float(string[5])
			if (z_SDSS >= z_min) & (z_SDSS < z_max) & (float(string[3]) == 1):
				if (float(string[2]) < mag_limit) & (float(string[1]) > flux_limit):
					total += 1
					if float(string[6]) > 0:
						found += 1
	if found != 0: sdss_completeness_corr = total/found
	else: sdss_completeness_corr = 1
	print("Totale = {0} / Con massa = {1} -> Shen_corr = {2:.2f}".format(total,found,sdss_completeness_corr))
	return sdss_completeness_corr
def ID_compl_calc(flux_limit, mag_limit):
	''' Calcola la frazione di oggetti presenti nella CLASS vs quelli con un identificativo (quindi un redshift)
	'''
	found = 0
	total = 0
	with open('sdss_in_class.txt') as sdss:
		next(sdss)
		for line in sdss:
			string = line.split()
			if (float(string[1]) >= flux_limit) & (float(string[2]) <= mag_limit) & (float(string[2]) > 0):
				total += 1
				if float(string[4]) > 0:
					found += 1
	if found != 0: ID_compl_corr = total/found
	else: ID_compl_corr = 1
	print("Totale = {} / Con ID = {} -> ID_corr = {}".format(total,found,ID_compl_corr))
	return ID_compl_corr
def flux_mag_limits(z):
	dist_z = ned_calc(z)
	dist4_5 = ned_calc(4.5)
	dist_ratio = dist4_5/dist_z
	# Calcolo dei limiti di luminosita (L_nu * nu) a cui siamo sensibili a z=4.5, i limiti sono applicati poi a tutti i bin
	magn4_5 = 21.5
	radio_limit = 30*dist_ratio**2*((1+4.5)/(1+z))**(k_corr)
	mag_limit = magn4_5-5*math.log10(dist_ratio)-2.5*k_corr*math.log10((1+4.5)/(1+z)) -2.5*alpha*math.log10(1400/6580)
	return radio_limit,mag_limit,

bin_width = 0.5
z_min = 1.5
z_max = 4
alpha = 0.44 # spectral index
k_corr = alpha-1
mass_limit = 9
z_bin =  [1.5,1.62,1.875,2,2.2,2.4,2.5,2.75,3,3.5,4] # array contenente z_min e z_max di ogni bin
count = np.zeros(len(z_bin)-1)
dist4_5 = ned_calc(4.5) # distanza di luminosità a z=4.5
z,radio_flux,mag	= [],[],[]

'''m_CIV_media = []
m_noC_media =[]
with open('shen_radio.txt') as fp:
	#Salva tutti gli oggetti con massa calcolata da Shen (e mag non nulla)
	lines = fp.readlines()[1:]
	for line in lines:
		string = line.split()
		if (float(string[6]) > 0):
			if (string[6] != string[4]):
				m_noC_media.append(float(string[4]))
				m_CIV_media.append(float(string[6]))
offset = np.average(np.asarray(m_noC_media) - np.asarray(m_CIV_media))'''

with open('sdss_in_class.txt') as fp:
	#Salva tutti gli oggetti con massa calcolata da Shen (e mag non nulla)
	lines = fp.readlines()[1:]
	for line in lines:
		string = line.split()
		if (float(string[9]) >= mass_limit) and (float(string[2]) > 0):
			radio_flux.append(	float(string[1]))
			mag.append(			float(string[2]))
			z.append(			float(string[4]))
		'''if float(string[6]) != float(string[9]):
			if (float(string[6]) - offset >= mass_limit) and (float(string[2]) > 0):
				name.append(		string[0])
				radio_flux.append(	float(string[1]))
				mag.append(			float(string[2]))
				flag_class.append(	float(string[3]))
				z.append(			float(string[4]))
				m_shen.append(		float(string[6]) - offset)
		else:
			if (float(string[6]) >= mass_limit) and (float(string[2]) > 0):
				name.append(		string[0])
				radio_flux.append(	float(string[1]))
				mag.append(			float(string[2]))
				flag_class.append(	float(string[3]))
				z.append(			float(string[4]))
				m_shen.append(		float(string[6]))'''

'''with open('shen_radio.txt') as fp:
	#Salva tutti gli oggetti con massa calcolata da Shen (e mag non nulla)
	lines = fp.readlines()[1:]
	for line in lines:
		string = line.split()
		if float(string[6]) == float(string[4]):
			if (float(string[4]) > 9) and (float(string[2]) > 0):
				name.append(		string[0])
				radio_flux.append(	float(string[2]))
				mag.append(			float(string[3]))
				z.append(			float(string[1]))
				m_shen.append(		float(string[4]))
		else:
			if (float(string[4])>0) and ((float(string[4]) - offset) > 9) and (float(string[2]) > 0):
				name.append(		string[0])
				radio_flux.append(	float(string[2]))
				mag.append(			float(string[3]))
				z.append(			float(string[1]))
				m_shen.append(		float(string[4]) - offset)'''

for i in range(len(z_bin)-1):
	condition = [(np.asarray(z) > z_bin[i]) & (np.asarray(z) < z_bin[i+1])]	#seleziona i redshift compresi nell' i-esimo bin
	choselist = [z]
	selected_z = np.select(condition, choselist)
	print("z= {} - {}".format(z_bin[i],z_bin[i+1]))
	mean_z = (z_bin[i+1] + z_bin[i])*0.5			#redshift medio del bin

	for j in range(len(selected_z)):				# applica i tagli in luminosità radio e ottica
		if (selected_z[j] > 0):
			flux_limit, mag_limit = flux_mag_limits(selected_z[j])
			if (radio_flux[j] >= flux_limit) & (mag[j] <= mag_limit):
				count[i] += 1

	print("Conteggi reali = {}".format(count[i]))

	sdss_completeness_corr = sdss_compl_calc(z_bin[i], z_bin[i+1])     # coefficiente di completezza SDSS/Shen (se Shen ha calcolato la massa di tutti gli oggetti presenti nella SDSS in un certo bin, coeff == 1)
	count[i] *= sdss_completeness_corr
	flux_limit, mag_limit = flux_mag_limits(mean_z)
	ID_compl_corr = ID_compl_calc(flux_limit, mag_limit) # moltiplica per un coef correttivo dovuto alla mancanza nel catalogo SDSS di alcuni oggetti CLASS (~40%)
	count[i] *= ID_compl_corr
	print("m limit = {}".format(mag_limit))
	print("radio limit = {}".format(flux_limit))

	print("Conteggi corretti = {}\n".format(count[i]))

write_counts(z_bin, count)
