# -*- coding: utf-8 -*-
"""
Diese Datei sollte nicht verändert werden und wird von uns gestellt und zurückgesetzt.

Skript testet das vortrainierte Modell


@author: Maurice Rohr
"""


from predict import predict_labels
from wettbewerb import load_references, save_predictions
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict given Model')
    parser.add_argument('--test_dir', action='store',type=str,default='../test/')
    parser.add_argument('--model_name', action='store',type=str,default='model.npy')
    parser.add_argument('--allow_fail',action='store_true',type=bool,default=False)
    args = parser.parse_args()
    
    # Importiere EEG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name
    ids, rec_channels, rec_data, sampling_frequencies, reference_systems, eeg_labels = load_references(args.test_dir) 
    predictions = list()
    start_time = time.time()
    for id,channels,data,fs,ref_system in zip(ids, rec_channels, rec_data, sampling_frequencies, reference_systems):
       
        try:
            _prediction = predict_labels(channels,data,fs,ref_system,model_name=args.model_name)
            _prediction["id"] = id
            predictions.append(_prediction)
        except:
            if args.allow_fail:
                raise
        
    pred_time = time.time()-start_time
    
    save_predictions(predictions) # speichert Prädiktion in CSV Datei
    print("Runtime",pred_time,"s")
