import csv
import numpy as np
import os
from wettbewerb import load_references, get_3montages
import mne
from scipy import signal as sig
import ruptures as rpt
import pywt
import matplotlib


# transform the time-domain signal to frequency domain and split it into 2 seconds epoch
def split_eeg_signal_into_frequency_domains(eeg_signal, sampling_rate):
    waveform_length = 2 * sampling_rate
    num_waveforms = len(eeg_signal) // waveform_length
    frequency_domains = []
    for i in range(num_waveforms):
        start_index = i * waveform_length
        end_index = start_index + waveform_length
        waveform = eeg_signal[start_index:end_index]

        # Continuous Wavelet Transform
        coefficients, _ = pywt.cwt(waveform, np.arange(1, 128), 'morl')

        frequency_domains.append(coefficients)
    frequency_domains = np.array(frequency_domains)
    return frequency_domains


# The same number of episodes of non-epileptic seizures were randomly selected as the number of episodes of all seizures
def random_select(arr, size):
    random_indices = np.random.choice(arr.shape[0], size=size, replace=False)

    # use the selected indices to slice the array
    selected_elements = arr[random_indices, :, :]

    return selected_elements


# pre-prosseing the dataset
def pre_processing(path):
    training_folder = path
    ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder)

    # label
    non_seizure_list = np.zeros((3, 1))
    seizure_list = np.ones((3, 1))
    for i, _ids in enumerate(ids):
        _fs = sampling_frequencies[i]
        _eeg_label = eeg_labels[i]
        start_time = _eeg_label[1]
        end_time = _eeg_label[2]
        new_montage, new_data, is_missing = get_3montages(channels[i], data[i])

        if _eeg_label[0] == 1:
            si = []
            non_si = []
            for j, signal_name in enumerate(new_montage):
                signal = new_data[j]
                time_ax = np.arange(new_data.shape[1]) / sampling_frequencies[i]
                signal_filter = mne.filter.filter_data(data=signal, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2,
                                                       verbose=False)
                _seizure = np.vstack((signal_filter, time_ax))
                # split seizure from a seizure signal
                bool_seizure = (_seizure[1] >= start_time) & (_seizure[1] <= end_time)
                bool_nseizure = (_seizure[1] < start_time) | (_seizure[1] > end_time)
                si.append(_seizure[0][bool_seizure])
                non_si.append(_seizure[0][bool_nseizure])

            si = np.asarray(si)
            non_si = np.asarray(non_si)
            seizure_list = np.concatenate((seizure_list, si), axis=1)
            non_seizure_list = np.concatenate((non_seizure_list, non_si), axis=1)

            seizure_list = np.asarray(seizure_list)
            non_seizure_list = np.asarray(non_seizure_list)
            # print('se: ',seizure_list.shape)


        else:
            non_si = []
            for j, signal_name in enumerate(new_montage):
                # split non seizure from a non seizure signal
                signal = new_data[j]
                signal_filter = mne.filter.filter_data(data=signal, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2,
                                                       verbose=False)
                non_si.append(signal_filter)
            non_si = np.asarray(non_si)
            non_seizure_list = np.concatenate((non_seizure_list, non_si), axis=1)
            non_seizure_list = np.asarray(non_seizure_list)


    waveform_seizure = []
    for i in np.arange(3):
        waveform = split_eeg_signal_into_frequency_domains(seizure_list[i], _fs)
        waveform_seizure.append(waveform)
    waveform_seizure = np.asarray(waveform_seizure)


    waveform_non_seizure = []
    for i in np.arange(3):
        waveform = split_eeg_signal_into_frequency_domains(seizure_list[i], _fs)
        waveform_selected = random_select(waveform, len(waveform_seizure[1]))
        waveform_non_seizure.append(waveform)

    waveform_non_seizure = np.asarray(waveform_seizure)


    return waveform_seizure, waveform_non_seizure





