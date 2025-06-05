# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:52:15 2021

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

filter_value = 9
save = False

Path = r'E:\AAVDJ.GluSnFR-S72A\iGluSnFR-S72A_LFD'
rec1 = pd.read_excel('{}/20210928_linescan1_recovery1AVG_50Hz_3pulses_2.5mMCa_dF_F0.xlsx'.format(Path))
rec2 = pd.read_excel('{}/20210928_linescan1_recovery2AVG_50Hz_3pulses_2.5mMCa_dF_F0.xlsx'.format(Path))


START = [0.019, 0.614, 1.021, 1.427, 1.825, 2.272, 2.639, 3.075, 3.478, 3.885]
STOP = [0.092, 0.687, 1.093, 1.5, 1.898, 2.345, 2.712, 3.148, 3.55, 3.958]


ep1, time1 = rec1.iloc[:,1], rec1['Time']
ep2, time2 = rec2.iloc[:,1], rec2['Time']

time2 = time2 + time1.iloc[-1] + (time1.iloc[-1] - time1.iloc[-2])

df = pd.concat([pd.DataFrame(np.ravel([ep1,ep2])), pd.DataFrame(np.ravel([time1,time2]))], axis=1)
df.columns = ['Trace', 'Time']


# pulse_start = np.ravel(np.where(df['Time'] >= 3.05))[0]
# pulse_stop = np.ravel(np.where(df['Time'] >= 3.15))[0]

# start = np.arange(df['Time'][pulse_start], df['Time'][pulse_start] + (0.4*2) +0.4, 0.4)
# stop = np.arange(df['Time'][pulse_stop], df['Time'][pulse_stop] + (0.4*2) + 0.4, 0.4)

indexes_start = [np.ravel(np.where(df['Time'] >= i))[0] for i in START]
indexes_stop = [np.ravel(np.where(df['Time'] >= i))[0] for i in STOP]

single_peak = [df['Trace'][i:j].values for i, j in zip(indexes_start, indexes_stop)]
single_peak_avg = np.mean(single_peak, axis=0)


start_burst = np.ravel(np.where(df['Time'] >= 4.2))[0]
end_burst = np.ravel(np.where(df['Time'] >= 4.7))[0]

burst = df['Trace'][start_burst : end_burst]
time_burst = df['Time'][start_burst : end_burst]


plt.figure()
plt.plot(df['Time'], df['Trace'], c = '#005172')
plt.plot(df['Time'], savgol_filter(df['Trace'], filter_value, 2), c = 'r')
for i, j in zip(START, STOP):
    plt.axvline(i, c = 'g')
    plt.axvline(j, c = 'g')


plt.figure()
[plt.plot(df['Time'][indexes_start[0]:indexes_stop[0]], single_peak[i], alpha = 0.2) for i in range(len(single_peak))]
plt.plot(df['Time'][indexes_start[0]:indexes_stop[0]], single_peak_avg, c = '#005172')
plt.plot(df['Time'][indexes_start[0]:indexes_stop[0]], savgol_filter(single_peak_avg, filter_value, 2), c = 'r')


plt.figure()
plt.plot(time_burst, burst, c = '#005172')
plt.plot(time_burst, savgol_filter(burst, filter_value, 2), c = 'r')

df_peaks = pd.concat([pd.DataFrame([single_peak[i] for i in range(len(single_peak))]).transpose(),
                        pd.DataFrame(single_peak_avg, columns = ['Average']),
                        df['Time'][0 : indexes_stop[0] - indexes_start[0]],
                        pd.DataFrame(START, columns = ['Idx_start']),
                        pd.DataFrame(STOP, columns = ['Idx_stop'])], axis=1)

df_burst = pd.concat([burst, time_burst], axis=1)

if save == True:
    with pd.ExcelWriter('{}/20210928_linescan2_recovery_peaks_indexes_button5_dF_F0.xlsx'.format(Path)) as writer:
        df_peaks.to_excel(writer)
    
    with pd.ExcelWriter('{}/20210928_linescan2_recovery_burst_button5_dF_F0.xlsx'.format(Path)) as writer:
        df_burst.to_excel(writer)