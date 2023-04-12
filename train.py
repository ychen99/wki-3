# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese

"""


import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from wettbewerb import load_references
import mne
from scipy import signal as sig
import ruptures as rpt


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

onset_list_predict = []
onset_list = []

ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references('Y:/External Databases/TUH EEG Seizure Corpus/data_mat/test_mat_wki') # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name (meist fs=300 Hz)

for i in range(len(ids)):
    fs = sampling_frequencies[i]
    eeg_signals = data[i]
    eeg_label = eeg_labels[i]
    if eeg_label[0]:
        onset_list.append(eeg_label[1])
        for j, signal_name in enumerate(channels[i]):
            # Get one channel of EEG
            signal = eeg_signals[j]
            # Apply notch filter to cancel out supply frequency
            signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
            # Apply bandpass filter between 0.5Hz and 70Hz to remove some noise from the signal
            signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
            #Parameter erklären
            #Überkommentieren
            #Compute short time fourier transformation of the signal, signal_filterd = filtered signal of channel, fs = sampling frequency, nperseg = length of each segment
            # Output f= array of sample frequencies, t = array of segment times, Zxx = STFT of signal
            f, t, Zxx = sig.stft(signal_filter, fs, nperseg=fs * 3)
            # Calculate step size of frequency
            df = f[1] - f[0]
            #Compute energy of the parts based on real and imaginary values of STFT
            E_Zxx = np.sum(Zxx.real ** 2 + Zxx.imag ** 2, axis=0) * df

            # Check if new array per patient has been created
            if j == 0:
                # Initialize array with energy signal of first channel
                E_array = np.array(E_Zxx)
            else:
                # Append energy signal of channel to the array (stack it)
                E_array = np.vstack((E_array, np.array(E_Zxx)))

        #Sum up energy of all channels
        E_total = np.sum(E_array, axis=0)
        # Get the index for the time at which the energy is maximum
        max_index = E_total.argmax()

        # Compute changepoints of the summed up signal
        # Check if index is at the beginning of the signal (since we are choosing a changepoint before the maximum method will cuase an error in that case
        if max_index == 0:
            # In this case seizure onset is at the beginning of the signal
            onset_list_predict.append(0.0)
        else:
            # Compute changepoints of the signal with method from ruptures package
            # Setup linearly penalized segmentation method to detect changepoints in signal with rbf cost function
            algo = rpt.Pelt(model="rbf").fit(E_total)
            # Get sorted list of changepoints, pen = penalty value
            result = algo.predict(pen=10)
            #Indices are shifted by one so subtract one
            result1 = np.asarray(result) - 1
            # Get the changepoints before the maximum
            result_red = result1[result1 < max_index]
            # Check if changepoint was found
            if len(result_red)<1:
                # If no changepoint was found, seizure onset is likely to be close to maximum
                print('no element')
                onset_index = max_index
            else:
                # Choose the changepoint which is closest to the maximum = seizure Onset
                onset_index = result_red[-1]
            # Append seizure onset to list
            onset_list_predict.append(t[onset_index])

# Compute absolute error between compute seizure onset and real onset based on doctor annotations
prediction_error = np.abs(np.asarray(onset_list_predict) - np.asarray(onset_list))

# Plot error per patient
plt.figure(1)
plt.scatter(np.arange(1, len(prediction_error)+1),prediction_error)
#plt.hlines(10, 0, len(prediction_error)+1, colors='red')
plt.ylabel('Error in s')
plt.xlabel('Patients')
plt.show()