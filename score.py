import os
import sys
import pandas as pd
import argparse

# TODO replace by new scoring code


def score(test_dir='../test/'):
        
    if not os.path.exists("PREDICTIONS.csv"):
        sys.exit("Es gibt keine Predictions")  

    if  not os.path.exists(os.path.join(test_dir, "REFERENCE.csv")):
        sys.exit("Es gibt keine Ground Truth")  

    df_pred = pd.read_csv("PREDICTIONS.csv")   # Klassifikationen
    df_gt = pd.read_csv(os.path.join(test_dir,"REFERENCE.csv"), sep=';',header=None)  # Wahrheit

    N_files = df_gt.shape[0]    # Anzahl an Datenpunkten
    N_seizures = 0
    
    ONSET_PENALTY = 60 # Sekunden, falls Anfall nicht erkannt oder größer als Penalty, wird Penalty gewertet

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
                delta_t_onset = max(abs(pred_onset-gt_onset),ONSET_PENALTY)
                delta_t_offset = max(abs(pred_offset-gt_offset),ONSET_PENALTY)
                
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
    confusion_matrix = [TP,FN,FP,TN]
    
    
    return F1,sensitivity,PPV,detection_error_onset,detection_error_offset,confusion_matrix

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Predict given Model')
    parser.add_argument('--test_dir', action='store',type=str,default='../test/')
    args = parser.parse_args()
    F1,sensitivity,PPV,detection_error_onset,detection_error_offset,confusion_matrix = score(args.test_dir)
    print("F1:",F1,"\t Latenz:",detection_error_onset)


