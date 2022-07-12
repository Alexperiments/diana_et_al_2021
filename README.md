# Source code of Diana et. al 2022

**Paper Title: The evolution of the heaviest supermassive black holes in jetted AGNs**

**Authors: A. Diana, A. Caccianiga, L. Ighina, S. Belladitta, A. Moretti, R. Della Ceca**

[ArXiv preprint](https://arxiv.org/pdf/2202.03444.pdf)

[Here](https://linktr.ee/dianaalessandro) you can find the poster presented at the AGN XIV congress in Florance (2022)

## Steps to reproduce the paper results

1.   `utils.selection()` selects the blazars in the sample CLASS within the magnitude and radio limits:
- $\lambda L_\lambda(1350\ Å) \geq 10^{46} erg\ s^{-1}$
- $P_{5 GHz} \geq 10^{27} W\ Hz^{-1}$

The script list all the blazar candidates in `data/selection.txt`.
The script also checks if the blazar has a saved SDSS spectrum in the *spectra_fits* folder and add this information to `data/selection.txt`.

2. `txtfits_to_hdf.py` reads `parameters.txt`, loads the data of each blazar (`.fits` or `.txt`) in `spectra_fits` and pre-elaborates the spectra.
The script brings the spectrum to the rest frame according to the tabulated redshift, trim the rest-frame wavelenghts in order to be centered around the CIV line.
The spectrum of each object (including the inverce variance) is then saved in `objects_pickles/OBJ_NAME/` as `spectrum.pkl` and the photometric, radio and redshift data as `info.pkl`.

3. `point_and_interval.py` gives a graphical interface for allowing the selection of the continuum intervals and the possible masks. The continuum selection mode is activated by pressing `c`.
Click 4 times to select two regions with high S/N around the CIV line.
The script automatically performs the continuum fit with a line.
The mask selection is activated by pressing `m`.
Each mask is selected with two clicks around the region to be masked (first click must be at the left of the second one).
In both selection modes it is possible to cancel the last selection by pressing `canc`.
By pressing `r` the program exits the selection mode and the user can freely click on the plot.
When satisfied press `f4` to save the intervals in `objects_pickles/OBJ_NAME/pre_fit.pkl`.

4. `parameters_estimation.py` provides a second graphical interface to select the best gaussian fit for the CIV line.
The script reads the `spectrum.pkl` and `pre_fit.pkl` of the blazar and shows a three plots of the continuum-subtracted masked spectrum, each one with a different number of gaussian components (1, 2 or 3). Then the user evaluates by eye the best residual and select the best fit by first closing the plot (`alt+f4`) and then by inserting the number of components when prompted by the script.
The script saves the parameters calculated with the fitting procedure (i.e. line dispersion, FWHM, line luminosity, etc...) in `objects_pickles/OBJ_NAME/parameters.pkl`.
The scripts also performs the uncertainties estimate on those parameters by perturbing the true spectrum with a gaussian noise according to the inverse variance (when present, or using the noise around the line).

5. `utils.save_parameters_list()` creates a global file containing all the tabulated and estimated parameters with their uncertainties and saves it as `data/parameters.pkl`.

6. `density_estimate.py` reads the `parameters.pkl` file and counts the blazars with a massa greater than $10^9 M_\odot$. 
Then calculates the correction coefficients to account for the missing identifications and the missing spectra.
Finally, it calculates the density of the blazars as a function of redshift and the density of the radio-loud AGNs, assuming a certain Lorentz bulk factor $\Gamma$, with their uncertainties (see *Density uncertainties* for further details).
The script then prints the "numbers" of the sample in LaTeX table format (Tab. 1 and Tab. 4), plots the density and the RL fraction figures (Figs. 4 and 5) and saves them in `paper_plots` and finally prints the fit parameters of the broken power-law used to fit the RL densities.

7. In order to obtain the other relevant information present in the paper one can use the various scripts in `utils.py`:
- `utils.stamp_plot_diskVSse`: to obtain the comparison plot between disc masses and single epoch masses (saved in `paper_plots`).
- `utils.stamp_plot_LCIV_1350`: to plot the comparison between the luminosities of our blazar sample and the luminosities of randomly oriented AGNs in Shen et al. (2011), saved in `paper_plots`.
- `utils.print_C19_table`: to print the LaTeX table containing info of the C19 sample.
- `utils.get_luminosity_limits`: to print the limiting luminosities of our selection process.

## Density uncertainties

The density uncertainties estimate is performed as follows:
For each bin we have many blazars with a best-fit value of the mass. For each of these objects the script extracts `N` values for the mass from a gaussian distribution centred around the true value with a std. dev. $0.3\ dex$ (intrinsic scatter of the Vestergaard & Peterson 06 relation).
Then, it counts the `i` objects with a mass above $10^9\ M_\odot$ and extracts one value `P` from a Poisson distribution with $\lambda=\text{i}$.
It then calculates two mock correction coefficients (`c1` and `c2`) from two binomial distributions assuming $p=1/C_{id}$ and $p=1/C_{spect}$, respectively.
`P`, `c1` and `c2` are then multiplied to obtain a mock blazar density point, accounting for the uncertainties on the correction coefficients.
This procedure is performed $10^5$ times, in order to obtain a distribution for the blazar density at each redshift bin. 

The distributions are mainly Poissonian but with some perturbation, due to the coefficients uncertainties, and shifted with respect to the best value due to the relative abundance of objects with a mass slightly above or slightly below the $10^9\ M_\odot$ threshold.

The $1 \sigma$ interval is then calculated as the interval between the 16-th quantile and the 84-th quantile.

## Contact me
dianaalessandro96@gmail.com