# imports
import pandas as pd
import ast
import numpy as np
from scipy.signal import resample, butter, filtfilt
import matplotlib.pyplot as plt

# target resampling rate
TARGET_LENGTH = 625

# function for filtering
def bandpass_filter(signal, fs=250, lowcut=1, highcut=30, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)  # zero-phase filtering
    return filtered

# load the data (needs to be preconverted in step1.py file)
data = pd.read_pickle("processed_full_dataset.pkl")
print(data.shape)

data = data[data["ECG_Transition"] == False].reset_index(drop=True)
print(data["ECG_SR"].unique())
raw_signals = data["ECG_Data"].values
sampling_rates = data["ECG_SR"].values

processed_signals = []

for sig, sr in zip(raw_signals, sampling_rates):

    if sr in [125, 200]:
        sig = resample(sig, 625)

    if len(sig) != 625:
        print("Unexpected length:", len(sig))
        continue

    processed_signals.append(sig)

signals = np.array(processed_signals)

# filtering - band pass filter, dc mean, normalization
filtered_signals = []

for sig in signals:
    sig = bandpass_filter(sig)
    sig = sig - np.mean(sig)
    sig = sig / np.std(sig)
    filtered_signals.append(sig)

signals = np.array(filtered_signals)

# plotting for sanity check
# sig = data["ECG_Resampled"].iloc[2]
# filtered = data["ECG_Normalized"].iloc[2]

# plt.plot(sig, alpha=0.5, label="Raw")
# plt.plot(filtered, label="Filtered")
# plt.legend()
# plt.show()

X = signals[:, np.newaxis, :]
np.save("ecg_dataset_full.npy", X.astype(np.float32))