from matplotlib import rcParams

ALPHA_O = 0.44
ALPHA_R = 0
TRIM_INF = 1200
TRIM_SUP = 1800

N_MONTECARLO = 100

BOLOMETRIC_CORRECTION = 4.5

SHEN_FIT = 'B'

SDSS_IN_CLASS_FILE = 'data/class_in_sdss.txt'
PARAMETERS_FILE = 'data/parameters.pkl'
SOURCE_FOLDER = 'spectra_fits/'
OUT_FOLDER = 'objects_pickles/'
SELECTION_FILE = 'data/selection.txt'
PREPARATION_PLOTS = 'pre_fit_plots/'
FITTED_PLOTS = 'fit_plots/'
PAPER_PLOTS = 'paper_plots/'
C19_TEX = 'data/C19_table.tex'
TABLE_TEX = 'data/full_table.tex'

def default_plot_settings():
    '''Impostazioni di default per i plot destinati al paper.
    '''
    rcParams["figure.autolayout"] = True
    rcParams['savefig.bbox'] = 'tight'
    rcParams["font.family"] = 'serif'
    rcParams["mathtext.fontset"] = "dejavuserif"
    rcParams['figure.figsize'] = [5.5, 5]
    rcParams['font.size'] = 16
    rcParams['legend.fontsize'] = 'large'
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.major.size'] = 10
    rcParams['ytick.major.size'] = 10
    rcParams['xtick.minor.size'] = 5
    rcParams['ytick.minor.size'] = 5
    rcParams['xtick.minor.visible'] = True
    rcParams['ytick.minor.visible'] = True
    rcParams['xtick.major.pad'] = 5
    rcParams['ytick.major.pad'] = 5
    rcParams['xtick.top'] = True
    rcParams['ytick.right'] = True


def poster_plot_settings():
    '''Impostazioni di default per i plot destinati al paper.
    '''
    rcParams["figure.autolayout"] = True
    rcParams['savefig.bbox'] = 'tight'
    rcParams['figure.figsize'] = [5.5, 5]
    rcParams['font.size'] = 16
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.major.size'] = 0
    rcParams['ytick.major.size'] = 0
    rcParams['xtick.minor.size'] = 0
    rcParams['ytick.minor.size'] = 0
    rcParams['xtick.major.pad'] = 5
    rcParams['ytick.major.pad'] = 5