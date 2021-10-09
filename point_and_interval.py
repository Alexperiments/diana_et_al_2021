import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
import os
import config
import h5py


class Intervals_collector:
    def __init__(self, fig):
        self.continuum = int_dict['continuum']
        self.masks = []
        self.cid1 = fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.cid2 = fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid2 = fig.canvas.mpl_connect('close_event', self.on_close)
        self.continuum_mode = False
        self.masks_mode = False
        self.fig = fig
        self.ax = fig.gca()
        self.spans = []

    def on_key(self, event):
        if event.key == 'c':
            print("Seleziona intervalli per il continuo (max 4 punti); 'canc' per eliminare ultima selezione")
            self.continuum_mode = True
            self.masks_mode = False
        elif event.key == 'm':
            print("Seleziona coppie di punti per mascherare un intervallo; 'canc' per eliminare ultima selezione")
            self.continuum_mode = False
            self.masks_mode = True
        elif event.key == 'r':
            print("Premere 'c' per il continuo; 'm' per le maschere; 'f4' per terminare e salvare")
            self.continuum_mode = False
            self.masks_mode = False
        elif event.key == 'delete':
            try:
                if self.continuum_mode:
                    self.continuum.pop()
                    del self.ax.lines[1:]
                    self.update()
                elif self.masks_mode:
                    self.masks.pop()
                    self.masks.pop()
                    self.spans[-1].remove()
                    del self.spans[-1]
                    self.fig.canvas.draw()
                else: raise
            except(IndexError):
                print("Lista vuota!")
                pass
        elif event.key == 'n':
            del self.ax.lines[1:]
            int_dict['continuum'] = []
            self.continuum = []
            self.update()
        elif event.key == 'f4':
            if (len(self.continuum) == 4) and (len(self.masks)%2 == 0):
                int_dict['continuum'] = str(self.continuum)
                int_dict['masks'] = str(self.masks)
                name = int_dict['name'] + '.png'
                plt.savefig(config.PREPARATION_PLOTS+name, dpi=300)
                plt.close(self.fig)
            else:
                print(
                f"Punti continuo: {len(self.continuum)}/4\tPunti maschera: {len(self.masks)}"
                )

    def on_click(self, event):
        if self.continuum_mode:
            if len(self.continuum) < 4:
                self.continuum.append(event.xdata)
            else: print("Ho già 4 punti per il continuo, rimuoverne qualcuno!")
        elif self.masks_mode: self.masks.append(event.xdata)
        self.update()

    def on_close(self, event):
        del self.continuum
        del self.masks

    def update(self):
        for line in self.continuum:
            self.ax.axvline(line, color='green', ls='--')
        if (len(self.masks)%2 == 0) & (len(self.masks) != 0):
            self.spans.append(self.ax.axvspan(self.masks[-2], self.masks[-1], alpha=0.5, color='gray'))
        if len(self.continuum) == 4:
            m, q = continuum_fit(obj_spectra, self.continuum)
            int_dict['m'] = m
            int_dict['q'] = q
            x_bin = np.arange(self.continuum[0], self.continuum[3], 1)
            self.ax.plot(x_bin, q + m*x_bin, color='red')
        self.fig.canvas.draw()


def continuum_fit(spectrum_df, int):
    mask = (
        ((spectrum_df['lambda'] >= int[0]) &
        (spectrum_df['lambda'] < int[1])) |
        ((spectrum_df['lambda'] >= int[2]) &
        (spectrum_df['lambda'] < int[3]))
    )
    continuum_df = spectrum_df[mask]
    m, q = np.polyfit(continuum_df['lambda'], continuum_df['flux'], 1)
    return m, q

for obj_name in os.listdir(config.HDF_FOLDER):
    file_path = os.path.join(config.HDF_FOLDER, obj_name)
    print(obj_name)
    obj_spectra = pd.read_hdf(file_path, key='spectrum')
    obj_info = pd.read_hdf(file_path, key='info')

    int_dict = {
        'name': obj_name,
        'continuum': [1435,1455,1690,1710],
    }
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.plot(obj_spectra['lambda'], obj_spectra['flux'], color='black', lw=0.5)
    ax.set_title(obj_name)
    for xi in int_dict['continuum']:
        ax.axvline(xi, color='green', ls='--')
    m, q = continuum_fit(obj_spectra, int_dict['continuum'])
    int_dict['m'] = m
    int_dict['q'] = q
    x_bin = np.arange(int_dict['continuum'][0], int_dict['continuum'][3], 1)
    ax.plot(x_bin, q + m*x_bin, color='red')
    int = Intervals_collector(fig)
    plt.show()

    print(int_dict)
    df = pd.DataFrame(int_dict, index=[0])
    df.reset_index().to_hdf(
        os.path.join(config.HDF_FOLDER,obj_name), key="pre_fit", mode='a'
    )
