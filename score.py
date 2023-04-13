import os
import sys
import pandas as pd
import argparse
from typing import Tuple,List


def score(test_dir : str='../test/') -> Tuple[float,float,float,float,float,float,List[int]]:
    """
    Berechnet relevante Metriken des Wettbewerbs, sowie weitere Metriken
    Parameters
    ----------
    folder : str, optional
        Ort der Testdaten. Default Wert '../test'.

    Returns
    -------
    performance_metric : float
        Metrik wird für das Ranking verwendet. MAE der Onset Detektion mit Strafterm für Fehlklassifikation
    F1 : float
        F1-Score der Seizure-Klassifikation (Seizure present = postive Klasse)
    sensitivity :  float
        Sensitivität der Seizure Klassifikation
    PPV : float
        Positive Predictive Value der Seizure Klassifikation
    detection_error_onset : float
        Mittlerer Absoluter Fehler der Onset Detektion (mit oberem Limit pro Aufnahme)
    detection_error_offset : float
        Mittlerer Absoluter Fehler der Offset Detektion (mit oberem Limit pro Aufnahme)
    confusion_matrix : List[int]
        Confusion Matrix der Seizure Klassifikation [TP,FN,FP,TN]
    """
        
    if not os.path.exists("PREDICTIONS.csv"):
        sys.exit("Es gibt keine Predictions")  

    if  not os.path.exists(os.path.join(test_dir, "REFERENCE.csv")):
        sys.exit("Es gibt keine Ground Truth")  

    df_pred = pd.read_csv("PREDICTIONS.csv")   # Klassifikationen
    df_gt = pd.read_csv(os.path.join(test_dir,"REFERENCE.csv"), sep=',',header=None)  # Wahrheit

    N_files = df_gt.shape[0]    # Anzahl an Datenpunkten
    N_seizures = 0
    
    ONSET_PENALTY = 60 # Sekunden, falls Anfall nicht erkannt oder größer als Penalty, wird Penalty gewertet
    FALSE_CLASSIFICATION_PENALTY = 60 # Sekunden, falls Anfall erkannt wird, obwohl keiner vorliegt, werden Strafsekunden vergeben

    ## für F1-Score basierend auf Anfall erkannt / nicht erkannt
    TP = 0  # Richtig Positive
    TN = 0  # Richtig Negative
    FP = 0  # Falsch Positive
    FN = 0  # Falsch Negative
    
    detection_error_onset = 0 # Durchschnittliche Latenz bei der Onset Detektion
    detection_error_offset = 0 # Durchschnittliche Latenz bei der Offset Detektion
    

    for i in range(N_files):
        gt_name = df_gt[0][i]
        gt_seizure_present = df_gt[1][i]
        gt_onset = df_gt[2][i]
        gt_offset = df_gt[3][i]

        pred_indx = df_pred[df_pred['id']==gt_name].index.values

        if not pred_indx.size:
            print("Prediktion für " + gt_name + " fehlt, nehme \"kein Anfall\" an.")
            pred_seizure_present = 0
            pred_seizure_confidence = 0.0
            pred_onset = -1
            pred_onset_confidence = 0.0
            pred_offset = -1
            pred_offset_confidence = 0.0
        else:
            pred_indx = pred_indx[0]
            pred_seizure_present = df_pred['seizure_present'][pred_indx]
            pred_seizure_confidence = df_pred['seizure_confidence'][pred_indx]
            pred_onset = df_pred['onset'][pred_indx]
            pred_onset_confidence = df_pred['onset_confidence'][pred_indx]
            pred_offset = df_pred['offset'][pred_indx]
            pred_offset_confidence = df_pred['offset_confidence'][pred_indx]
        
        if gt_seizure_present:
            N_seizures += 1
            if pred_seizure_present == 0:
                delta_t_offset = ONSET_PENALTY
                delta_t_onset = ONSET_PENALTY
            else:    
                delta_t_onset = min(abs(pred_onset-gt_onset),ONSET_PENALTY)
                delta_t_offset = min(abs(pred_offset-gt_offset),ONSET_PENALTY)
                
            detection_error_offset += delta_t_offset
            detection_error_onset += delta_t_onset
            
  
        TP += int(gt_seizure_present and pred_seizure_present)
        TN += int((not gt_seizure_present) and (not pred_seizure_present))
        FN += int(gt_seizure_present and (not pred_seizure_present))
        FP += int((not gt_seizure_present) and  pred_seizure_present)
        
        
    
    sensitivity = TP/(TP+FN)
    PPV = TP/(TP+FP)
    F1 = 2*sensitivity*PPV/(sensitivity+PPV)
    
    detection_error_offset = detection_error_offset / N_seizures
    detection_error_onset = detection_error_onset / N_seizures
    
    # Finale Metrik des Wettbewerbs besteht aus dem Absoluten Onset-Fehler und den Strafsekunden für fälschlich erkannte Anfälle
    performance_metric = detection_error_onset + (FP/(FP+TN))*FALSE_CLASSIFICATION_PENALTY*(1-N_seizures/N_files)
    
    confusion_matrix = [TP,FN,FP,TN]
    
    
    return performance_metric,F1,sensitivity,PPV,detection_error_onset,detection_error_offset,confusion_matrix

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Predict given Model')
    parser.add_argument('--test_dir', action='store',type=str,default='../test/')
    args = parser.parse_args()
    performance_metric,F1,sensitivity,PPV,detection_error_onset,detection_error_offset,confusion_matrix = score(args.test_dir)
    print("WKI Metrik:", performance_metric,"\t F1:",F1,"\t Latenz:",detection_error_onset)


