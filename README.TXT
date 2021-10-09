

1) Lo script utils.selection() seleziona gli oggetti che rispettano i limiti del nostro sample, controlla quanti non hanno uno spettro nella cartella "spectra_fits" e poi salva la selezione nel file data/selection.txt

2) Lo script txtfits_to_hdf.py prepara gli spettri, portandoli nel restframe e riducendo il range di lunghezze d'onda, accorpa i dati dello spettro con le info fotometriche, radio e il redshift, e salva tutto in un file compresso hdf per ogni oggetti, contenente 2 tabella. La prima tabella "spectrum", contiene lunghezze d'onda, flusso e ivar rest-frame. La seconda tabella "info" contiente le informazioni ausiliare.

3) point_and_interval.py fornisce l'interfaccia grafica per la selezione degli intervalli del continuo e le maschere personalizzate da applicare agli spettri. 
