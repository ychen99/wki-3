# -*- coding: utf-8 -*-
"""
Diese Datei sollte nicht verändert werden und wird von uns gestellt und zurückgesetzt.

Funktionen zum Laden und Speichern der Dateien
"""
__author__ = "Maurice Rohr und Dirk Schweickard"

from typing import List, Tuple, Dict, Any
import csv
import scipy.io as sio
import numpy as np
import os


### Achtung! Diese Funktion nicht veraendern.
# TODO Passe an an neues Epilepsie Daten
# TODO resampling überdenken
def load_references(folder: str = '../training') -> Tuple[List[str], List[List[str]],
                                                          List[np.ndarray],  List[float],
                                                          List[str], List[Tuple[bool,float,float]]]:
    """
    Liest Referenzdaten aus .mat (Messdaten) und .csv (Label) Dateien ein.
    Parameters
    ----------
    folder : str, optional
        Ort der Trainingsdaten. Default Wert '../training'.

    Returns
    -------
    ids : List[str]
        Liste von ID der Aufnahmen
    channels : List[List[str]]
        Liste der vorhandenen Kanäle per Aufnahme
    data :  List[ndarray]
        Liste der Daten pro Aufnahme
    sampling_frequencies : List[float]
        Liste der Sampling-Frequenzen.
    reference_systems : List[str]
        Liste der Referenzsysteme. "LE", "AR", "Sz" (Zusatz-Information)
    """
    # Check Parameter
    
    
    
    assert isinstance(folder, str), "Parameter folder muss ein string sein aber {} gegeben".format(type(folder))
    assert os.path.exists(folder), 'Parameter folder existiert nicht!'
    # Initialisiere Listen für leads, labels und names
    ids: List[str] = []
    channels: List[List[str]] = []
    data: List[np.ndarray] = []
    sampling_frequencies: List[float] = []
    reference_systems: List[str] = []
    eeg_labels: List[Tuple[bool,float,float]]
    
    
    ecg_leads: List[np.ndarray] = []
    ecg_labels: List[str] = []
    ecg_names: List[str] = []
    # Setze sampling Frequenz
    fs: int = 300
    # Lade references Datei
    with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Iteriere über jede Zeile
        for row in csv_reader:
            # Lade MatLab Datei mit EKG lead and label
            data = sio.loadmat(os.path.join(folder, row[0] + '.mat'))
            ecg_leads.append(data['val'][0])
            ecg_labels.append(row[1])
            ecg_names.append(row[0])
    # Zeige an wie viele Daten geladen wurden
    print("{}\t Dateien wurden geladen.".format(len(ecg_leads)))
    return ids, channels, data, sampling_frequencies, reference_systems, eeg_labels




### Achtung! Diese Funktion nicht veraendern.
# TODO passen an an neues Thema
#predictions = {"id":id,"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
#                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
#                   "offset_confidence":offset_confidence}
def save_predictions(predictions: List[Dict[str,Any]], folder: str=None) -> None:
    """
    Funktion speichert the gegebenen predictions in eine CSV-Datei mit dem name PREDICTIONS.csv
    Parameters
    ----------
    predictions : List[Dict[str,Any]]
        Liste aus dictionaries. Jedes Dictionary enthält die Felder "id","seizure_present",
                "seizure_confidence","onset","onset_confidence","offset","offset_confidence"
	folder : str
		Speicherort der predictions
    Returns
    -------
    None.

    """    
	# Check Parameter
    assert isinstance(predictions, list), \
        "Parameter predictions muss eine Liste sein aber {} gegeben.".format(type(predictions))
    assert len(predictions) > 0, 'Parameter predictions muss eine nicht leere Liste sein.'
    assert isinstance(predictions[0], dict), \
        "Elemente der Liste predictions muss ein Dictionary sein aber {} gegeben.".format(type(predictions[0]))
	
    if folder==None:
        file = "PREDICTIONS.csv"
    else:
        file = os.path.join(folder, "PREDICTIONS.csv")
    # Check ob Datei schon existiert wenn ja loesche Datei
    if os.path.exists(file):
        os.remove(file)

    with open(file, mode='w', newline='') as predictions_file:

        # Init CSV writer um Datei zu beschreiben
        predictions_writer = csv.writer(predictions_file, delimiter=',')
        # Iteriere über jede prediction
        header=["id","seizure_present","seizure_confidence","onset","onset_confidence","offset","offset_confidence"]
        predictions_writer.writerow(header)
        for prediction in predictions:
            predictions_writer.writerow([prediction["id"], prediction["seizure_present"],
                                         prediction["seizure_confidence"],prediction["onset"],
                                         prediction["onset_confidence"],prediction["offset"],
                                         prediction["offset_confidence"]])
        # Gebe Info aus wie viele labels (predictions) gespeichert werden
        print("{}\t Labels wurden geschrieben.".format(len(predictions)))
        
# TODO schreibe Funktion zur Bildung der 3 Montagen Fp1-F2, Fp2-F4, C3-P3
def get_3montages(channels: List[str], data: np.ndarray) -> Tuple[List[str],np.ndarray,bool]:
    """
    Funktion berechnet die 3 Montagen Fp1-F2, Fp2-F4, C3-P3 aus den gegebenen Ableitungen (Montagen)
    zur selben Referenzelektrode. Falls nicht alle nötigen Elektroden vorhanden sind, wird das entsprechende Signal durch 0 ersetzt. 
    ----------
    channels : List[str]
        Namen der Kanäle z.B. Fp1, Cz, C3
	data : ndarray
		Daten der Kanäle
    Returns
    -------
    montages : List[str]
        Namen der Montagen ["Fp1-F2", "Fp2-F4", "C3-P3"]
    montage_data : ndarray
        Daten der Montagen
    montage_missing : bool
        1 , falls eine oder mehr Montagen fehlt, sonst 0

    """   
    pass


# TODO schreibe Funktion zur Bildung der 6 Montagen Fp1-F2, Fp2-F4, C3-P3, F3-C3, F4-C4, C4-P4
def get_montages(channels: List[str], data: np.ndarray) -> Tuple[List[str],np.ndarray,bool]:
    """
    Funktion berechnet die 6 Montagen Fp1-F2, Fp2-F4, C3-P3, F3-C3, F4-C4, C4-P4 aus den gegebenen Ableitungen (Montagen)
    zur selben Referenzelektrode. Falls nicht alle nötigen Elektroden vorhanden sind, wird das entsprechende Signal durch 0 ersetzt. 
    ----------
    channels : List[str]
        Namen der Kanäle z.B. Fp1, Cz, C3
	data : ndarray
		Daten der Kanäle
    Returns
    -------
    montages : List[str]
        Namen der Montagen ["Fp1-F2", "Fp2-F4", "C3-P3", "F3-C3", "F4-C4", "C4-P4"]
    montage_data : ndarray
        Daten der Montagen
    montage_missing : bool
        1 , falls eine oder mehr Montagen fehlt, sonst 0

    """  
    pass

