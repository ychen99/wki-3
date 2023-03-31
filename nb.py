import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy import signal as sig
import ruptures as rpt
# Load EEG recording of patient

# Load EEG data and the annotation file for one patient with gtka (generalisiert tonisch-klonische Anfall)
data = pd.read_pickle('Y:/Epilepsy_Data(Student)/GTKA/PAT2/Data.csv')
annotations = pd.read_pickle('Y:/Epilepsy_Data(Student)/GTKA/PAT2/Annotations.csv')
# Show annotations made by doctors
print(annotations)
# Set sampling frequency
sampling_freq = 256  # in Hz
# Define channels to use
ch_names = ['Fp1', 'Tp10', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
            'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Tp9', 'O2', 'FT9',
            'FT10', 'Sp1', 'Sp2']
# Set channel types
ch_types = ['eeg'] * 25
# Create an info object
info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sampling_freq)
print(info)
# Get Dataframe of selected channels defined above
data = data.transpose()
data_subset = data[ch_names].transpose()
# Convert Dataframe into numpy array
raw_data = pd.DataFrame.to_numpy(data_subset, dtype=np.float64)
# Convert signals(channels) into microvolt for better plotting
raw_data = raw_data*(10**-6)
#Create a raw MNE object with selected channels and info object
raw = mne.io.RawArray(raw_data, info)
# PLot raw object (EEG channels) with automatic scaling
raw.plot(n_channels=4, scalings='auto', title='EEG channels with auto-scaling',
         show=True, block=True)
# Compute and plot spectral density (averaged over all channels)
raw.compute_psd().plot(average=True)
# Notch filter to cancel out supply frequency
raw.copy().notch_filter(freqs=np.array([50,100])).compute_psd().plot(average=True)

#Initialize parameters

#Sample freq
fs = 256.0
# Onset times in s
onset_list = [305645 / 256, 2278321 / 256, 1978920 / 256, 716259 / 256, 1493180 / 256, 93818 / 256, 1912926 / 256,
              1467900 / 256, 889228 / 256, 1821956 / 256, 2314774 / 256, 1394146 / 256, 2163773 / 256, 1933448 / 256, 933751 / 256]
#List to store the onset predictions
onset_list_predict = []
# List of patients for the two classes
pat_list_mot = ['PAT1', 'PAT2', 'PAT3', 'PAT4', 'PAT5', 'PAT6', 'PAT7', 'PAT8'] #
pat_list_gtka = ['PAT2', 'PAT3', 'PAT4', 'PAT5', 'PAT6', 'PAT7', 'PAT8'] #
# List of used channels
signal_name_list = ['Fp1', 'Tp10', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                    'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Tp9', 'O2', 'FT9',
                    'FT10', 'Sp1', 'Sp2']

# Function for determining the seizure onset Assumption seizure is the section of the EEG signals with maximum energy. Calculate the energy of each channel with a rolling window using STFT. Sum up the individual signals obtained for all channels. Determine with the ruptures package changepoints in the received signal. Changepoint just before the peak of energy is declared as seizure onset

# Iterate over the two classes of seizures
for i in range(2):
    if i == 0:
        # Load list for patients with seizure type 'motorisch'
        pat_list = pat_list_mot
    else:
         # Load list for patients with seizure type 'tonisch-klonisch'
        pat_list = pat_list_gtka
    #Iterate over number of patients per class
    for idx, label in enumerate(pat_list):
        if i == 0:
            # Paths need to be adjusted
            # Get DataFrame of EEG signals for patient out of pat_list
            data_pat = pd.read_pickle('Y:/Epilepsy_Data(Student)/automotorisch, komplex motorisch/' + label + '/Data.csv')
            # Get annotations for patient out of pat_list
            annotations_pat = pd.read_pickle('Y:/Epilepsy_Data(Student)/automotorisch, komplex motorisch/' + label + '/Annotations.csv')
        else:
            # Paths need to be adjusted
            # Get DataFrame of EEG signals for patient out of pat_list
            data_pat = pd.read_pickle('Y:/Epilepsy_Data(Student)/GTKA/' + label + '/Data.csv')
            # Get annotations for patient out of pat_list
            annotations_pat = pd.read_pickle('Y:/Epilepsy_Data(Student)/GTKA/' + label + '/Annotations.csv')
        # Print patient number to track progress
        print(label)
        # Transpose Dataframe for accessing single channels
        data_pat = data_pat.transpose()
        #Iterate over all channels
        for idx_2, signal_name in enumerate(signal_name_list):
            # Get one channel of EEG and convert it into numpy array
            signal = data_pat[signal_name].to_numpy()
            # Apply notch filter to cancel out supply frequency
            signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
            # Apply bandpass filter between 0.5Hz and 70Hz to remove some noise from the signal
            signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
            #Parameter erklären
            #Überkommentieren
            #Compute short time fourier transformation of the signal, signal_filterd = filtered signal of channel, fs = sampling frequency, nperseg = length of each segment
            # Output f= array of sample frequencies, t = array of segment times, Zxx = STFT of signal
            f, t, Zxx = sig.stft(signal_filter, fs, nperseg=256 * 5)
            # Calculate step size of frequency
            df = f[1] - f[0]
            #Compute energy of the parts based on real and imaginary values of STFT
            E_Zxx = np.sum(Zxx.real ** 2 + Zxx.imag ** 2, axis=0) * df

            # Check if new array per patient has been created
            if idx_2 == 0:
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
            # Choose the changepoint which is closest to the maximum = seizure Onset
            onset_index = result_red[-1]
            # Append seizure onset to list
            onset_list_predict.append(t[onset_index])
# Function for determining the seizure onset Assumption seizure is the section of the EEG signals with maximum energy. Calculate the energy of each channel with a rolling window using squared amplitude. Sum up the individual signals obtained for all channels. Determine with the ruptures package changepoints in the received signal. Changepoint just before the peak of energy is declared as seizure onset

# Set window length to e.g. 5sec
window = 256 * 5
# Initiate list to store Energy of windows
E_Zxx = []
# Initiate list to store timepoints of the beginning of each windows
t = []

# Iterate over the two classes of seizures
for i in range(2):
    if i == 0:
        # Load list for patients with seizure type 'motorisch'
        pat_list = pat_list_mot
    else:
         # Load list for patients with seizure type 'tonisch-klonisch'
        pat_list = pat_list_gtka
    # Iterate over number of patients per class
    for idx, label in enumerate(pat_list):
        if i == 0:
            # Paths need to be adjusted
            # Get DataFrame of EEG signals for patient out of pat_list
            data_pat = pd.read_pickle('Y:/Epilepsy_Data(Student)/automotorisch, komplex motorisch/' + label + '/Data.csv')
            # Get annotations for patient out of pat_list
            annotations_pat = pd.read_pickle('Y:/Epilepsy_Data(Student)/automotorisch, komplex motorisch/' + label + '/Annotations.csv')
        else:
            # Paths need to be adjusted
            # Get DataFrame of EEG signals for patient out of pat_list
            data_pat = pd.read_pickle('Y:/Epilepsy_Data(Student)/GTKA/' + label + '/Data.csv')
            # Get annotations for patient out of pat_list
            annotations_pat = pd.read_pickle('Y:/Epilepsy_Data(Student)/GTKA/' + label + '/Annotations.csv')
        # Print patient number to track progress
        print(label)
        # Transpose Dataframe for accessing single channels
        data_pat = data_pat.transpose()
        #Iterate over all channels
        for idx_2, signal_name in enumerate(signal_name_list):
            # Get one channel of EEG and convert it into numpy array
            signal = data_pat[signal_name].to_numpy()
            # Apply notch filter to cancel out supply frequency
            signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
            # Apply bandpass filter between 0.5Hz and 70Hz to remove some noise from the signal
            signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
            # Compute Energy based of amplitude
            # clear lists which are used as intermediate storage per signal
            t.clear()
            E_Zxx.clear()
            # Compute number of windows n
            n = len(signal) // window
            # Iterate over all windows
            for k in range(n + 1):
                # Get part of the signal
                part = signal[(k * window):((k + 1) * window)]
                # Compute the energy of that part
                energy = np.sum(part ** 2) / fs
                # Append Energy to list
                E_Zxx.append(energy)
                #Append time to list
                t.append((k * window) / fs)
            # Check if new array per patient has been created
            if idx_2 == 0:
                # Initialize array with energy signal of first channel
                E_array = np.array(E_Zxx)
            else:
                # Append energy signal of channel to the array (stack it)
                E_array = np.vstack((E_array, np.array(E_Zxx)))
        # Convert list with time to array
        time = np.array(t)
        # Sum up energy of all channels
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
            # Choose the changepoint which is closest to the maximum = seizure Onset
            onset_index = result_red[-1]
            # Append seizure onset to list
            onset_list_predict.append(time[onset_index])

#Compute absolute error between compute seizure onset and real onset based on doctor annotations

prediction_error = np.abs(np.asarray(onset_list_predict) - np.asarray(onset_list))

#Plot error per patient

plt.figure(1)
plt.scatter(np.arange(1, 16),prediction_error)
#plt.hlines(10, 0, 16, colors='red')
plt.ylabel('Error in s')
plt.xlabel('Patients')

#Load data of patient and get single channel e.g. T4

data = pd.read_pickle('Y:/Epilepsy_Data(Student)/GTKA/PAT2/Data.csv')
annotations = pd.read_pickle('Y:/Epilepsy_Data(Student)/GTKA/PAT2/Annotations.csv')
data = data.transpose()
signal_T4 = data['T4'].to_numpy()

#Compute spectogram with matplotlib

plt.figure(2)
plt.specgram(signal_T4, Fs=256, cmap="rainbow")
plt.ylabel('Frequency in Hz')
plt.xlabel('Time in s')
plt.show()