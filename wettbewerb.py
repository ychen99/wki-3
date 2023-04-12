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
    eeg_labels: List[Tuple[bool,float,float]] = []
    
    
    # Lade references Datei
    with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Iteriere über jede Zeile
        for row in csv_reader:
            ids.append(row[0])
            eeg_labels.append((int(row[1]),float(row[2]),float(row[3])))
            # Lade MatLab Datei
            eeg_data = sio.loadmat(os.path.join(folder, row[0] + '.mat'),simplify_cells=True)
            ch_names = eeg_data.get('channels')
            ch_names = [x.strip(' ') for x in ch_names]
            channels.append(ch_names) 
            data.append(eeg_data.get('data'))
            sampling_frequencies.append(eeg_data.get('fs'))
            reference_systems.append(eeg_data.get('reference_system'))
    # Zeige an wie viele Daten geladen wurden
    print("{}\t Dateien wurden geladen.".format(len(ids)))
    return ids, channels, data, sampling_frequencies, reference_systems, eeg_labels




### Achtung! Diese Funktion nicht veraendern.
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
    montages = []
    _,m = np.shape(data)
    montage_data = np.zeros([3,m])
    montage_missing = 0
    try:
        montage_data[0,:] = data[channels.index('Fp1')] - data[channels.index('F3')]
        montages.append('Fp1-F3')
    except:
        montage_missing = 1
        montages.append('error')
    try:
        montage_data[1,:] = data[channels.index('Fp2')] - data[channels.index('F4')]
        montages.append('Fp2-F4')
    except:
        montage_missing = 1
        montages.append('error')
    try:
        montage_data[2,:] = data[channels.index('C3')] - data[channels.index('P3')]
        montages.append('C3-P3')
    except:
        montage_missing = 1
        montages.append('error')

    return (montages,montage_data,montage_missing)

def get_6montages(channels: List[str], data: np.ndarray) -> Tuple[List[str],np.ndarray,bool]:
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
    montages = []
    _,m = np.shape(data)
    montage_data = np.zeros([6,m])
    montage_missing = 0
    try:
        montage_data[0,:] = data[channels.index('Fp1')] - data[channels.index('F3')]
        montages.append('Fp1-F3')
    except:
        montage_missing = 1
        montages.append('error')
    try:
        montage_data[1,:] = data[channels.index('Fp2')] - data[channels.index('F4')]
        montages.append('Fp2-F4')
    except:
        montage_missing = 1
        montages.append('error')
    try:
        montage_data[2,:] = data[channels.index('C3')] - data[channels.index('P3')]
        montages.append('C3-P3')
    except:
        montage_missing = 1
        montages.append('error')
    try:
        montage_data[3,:] = data[channels.index('F3')] - data[channels.index('C3')]
        montages.append('F3-C3')
    except:
        montage_missing = 1
        montages.append('error')
    try:
        montage_data[4,:] = data[channels.index('F4')] - data[channels.index('C4')]
        montages.append('F4-C4')
    except:
        montage_missing = 1
        montages.append('error')
    try:
        montage_data[5,:] = data[channels.index('C4')] - data[channels.index('P4')]
        montages.append('C4-P4')
    except:
        montage_missing = 1
        montages.append('error')
    return (montages,montage_data,montage_missing)
