# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 03:07:23 2021

@author: Maurice
"""

from predict import predict_labels
from wettbewerb import load_references, save_predictions, EEGDataset
import argparse
import time
from score import score
import datetime
import os, csv

from wki_utilities import Database


'''
INPUT 
Data-Folder (where datasets located)
team_id or team_name
model information like parameters,name, binaryClassifier? (can also be return value of predictions)
which datasets to run (1 and 2, or 3 or 4)

'''

# ### INPUT #TODO delete
# data_folder='./../Datasets/'
# team_id = 1
# datasets_string = '1,2,3,4'
# datasets = [int(i) for i in datasets_string.split(',')]
# model_variants = 'binary' #'multi', 'both'
# model_name = 'ConvNet'
# parameter_dict = {'nr_epochs':20,'alpha':0.1,'n_layers':10} ## Extension
# output_file='error_out'


def wki_evaluate(data_folder,team_id,datasets_string,model_name,output_file='error_out',allow_fail=True,parameter_dict=None):

    datasets = [int(i) for i in datasets_string.split(',')]
    
    db = Database()
    nr_runs = db.get_nr_runs(team_id)
    if nr_runs==None:
        new_nr_runs=1
    else:
        new_nr_runs = nr_runs+1
    run_times = dict()
    run_successfull = dict()
    
    # TODO reconfigure output to file
    
    
    print("Got following parameters", "data_folder=",data_folder,", team_id=",team_id,", datasets_string=",datasets_string,", model_name=", model_name)       
        
    model_id = db.put_model(team_id,model_name,parameter_dict=None)
    
    for dataset_id in datasets:
        dataset_folder =  db.get_dataset_folder(dataset_id)
        dataset_name = os.path.basename(data_folder)  
        
        ### make predictions & measure time
        
        # Erstelle EEG Datensatz aus Ordner
        dataset = EEGDataset(dataset_folder)
        print(f"Teste Modell auf {len(dataset)} Aufnahmen von Dataset {dataset_name}")
        
        predictions = list()
        start_time = time.time()
        
        # Rufe Predict Methode für jedes Element (Aufnahme) aus dem Datensatz auf
        try:
            for item in dataset:
                id,channels,data,fs,ref_system,eeg_label = item
                try:
                    _prediction = predict_labels(channels,data,fs,ref_system,model_name=model_name)
                    _prediction["id"] = id
                    predictions.append(_prediction)
                except Exception as e:
                    if allow_fail:
                        raise
                    else : 
                        print("Exception ocurred for record",id)
                        print(e)
            print('Prediction SUCCESSFULL')
            print(f"(for model {model_name} and team_id {team_id})")
        except Exception as e:
            print(e)
            run_successfull[dataset_id]=False
        finally:
            run_times[dataset_id]=int(time.time()-start_time)
            print("Runtime",run_times[dataset_id],"s")
            
        if run_successfull[dataset_id]:
            save_predictions(predictions) # speichert Prädiktion in CSV Datei
            ### compute scores and save to database
            performance_metric,F1,sensitivity,PPV,detection_error_onset,detection_error_offset,confusion_matrix = score(os.path.join(data_folder, dataset_folder))

        
            db.put_scored_entry(dataset_id,team_id,new_nr_runs,performance_metric,F1,
                                sensitivity,PPV,detection_error_onset,detection_error_offset,
                                model_id,run_times[dataset_id],confusion_matrix)
        else:
            db.put_unscored_entry(dataset_id,team_id,model_id,run_times[dataset_id],output_file)
                
                
                

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Score based on Predictions and put in database')
    parser.add_argument('data_folder', action='store',type=str)
    parser.add_argument('team_id', action='store',type=int)
    parser.add_argument('--datasets', action='store',type=str,default='1,2')
    parser.add_argument('--allow_fail', action='store_true',default=False)
    parser.add_argument('--model_name', action='store',type=str,default='dummy') 
    parser.add_argument('--output_file', action='store',type=str,default='error_out')        

    args = parser.parse_args()
    
    wki_evaluate(args.data_folder,args.team_id,args.datasets,args.model_name,output_file=args.output_file,allow_fail=args.allow_fail,parameter_dict=None)               
