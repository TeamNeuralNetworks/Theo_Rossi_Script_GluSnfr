# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:51:45 2021

@author: Theo.ROSSI
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


Path = r'E:\AAVDJ.GluSnFR-S72A\Controls\LFD'
freq = '50Hz'
Linescan = '20210928_linescan2'
filter_value = 9
Save = False

###############################
### LOAD PROTOCOLE FILE  ######
###### AND SET LISTS ##########
###############################

file_protocole = pd.read_excel('{}\Protocole_LFD_Ephy\Protocole_LFD_last10pulses.xlsx'.format(Path))


EPISODE, TIME = [], []
START_EP, STOP_EP = [], []
for col in file_protocole.columns:
    if 'Ep' in col:
        episode = file_protocole[col]*100
        EPISODE.append(episode)
    elif 'Time' in col:
        time = file_protocole[col]
        TIME.append(time)
        start = file_protocole[col][0]
        START_EP.append(start)
        stop = np.ravel(file_protocole[col])[-1]
        STOP_EP.append(stop)



###############################
## LOAD BSL & RECOVERY FILES ##
###### MAKE FILES UNION #######
###############################
    
files = sorted(os.listdir(Path))

BSL_TRACES, BSL_AMPS = [],[]
RECOVERY1, RECOVERY2 = [],[]
BURST = []

for file in files:
    if freq in file:
        if Linescan in file:
            if 'dF_F0.xlsx' in file:
                if 'bsl' in file:
                    data = pd.read_excel('{}/{}'.format(Path, file))
                    BSL_TRACES.append(data)
                    
                elif 'recovery1' in file:
                    data = pd.read_excel('{}/{}'.format(Path, file))
                    trace = data.iloc[:,0]
                    time1 = data['Time']
                    RECOVERY1.append(trace)
                        
                elif 'recovery2' in file:
                    data = pd.read_excel('{}/{}'.format(Path, file))
                    trace = data.iloc[:,0]
                    time2 = data['Time']
                    RECOVERY2.append(trace)
                    
                elif 'burst' in file:
                    data = pd.read_excel('{}/{}'.format(Path, file))
                    trace = data.iloc[:,1]
                    time_burst = data['Time']
                    BURST.append(trace)
                


sampling_interval = time1.iloc[1]
time2 = time2 + (time1.iloc[-1] + sampling_interval)

full_traces = pd.concat((pd.DataFrame(RECOVERY1).transpose(), pd.DataFrame(RECOVERY2).transpose()), axis=0, ignore_index=True)
avg_full_trace = pd.concat((pd.DataFrame(np.mean(RECOVERY1, axis=0)), pd.DataFrame(np.mean(RECOVERY2, axis=0))), axis=0, ignore_index=True)
full_time = pd.concat((time1, time2), axis=0, ignore_index=True)

df_traces = pd.concat((full_traces, avg_full_trace, full_time), axis=1)
columns_names = [f'Ep{i+1}' for i in range(len(full_traces.columns))]
columns_names.append('Avg')
columns_names.append('Time')
df_traces.columns = columns_names
print(df_traces)


###############################
###### PEAKS EXTRACTION #######
######### LFD TRACES ##########
###############################

peak_time_list = [0.04, 0.635, 1.041, 1.449, 1.847, 2.293, 2.662, 3.098, 3.499, 3.898]
first_peak_idx = np.ravel(np.where(df_traces['Time'] <= peak_time_list[0]))[-1]
end_first_peak_idx = np.ravel(np.where(df_traces['Time'] >= df_traces['Time'][first_peak_idx]+0.06))[0]

PEAK_EP = []
for col in range(len(df_traces.columns)):
    if 'Ep' in df_traces.columns[col]:
        PEAKS = []
        fig, ax = plt.subplots(1,3,figsize=(14,4),tight_layout=True, sharey=True)
        
        time_bsl = BSL_TRACES[col]['Time']
        average = BSL_TRACES[col]['Average']
        trace = BSL_TRACES[col].drop(['Time', 'Average'], axis=1)
        
        [ax[0].plot(time_bsl, savgol_filter(trace[i], filter_value, 2), color='#253494', alpha=0.3) for i in range(len(trace.columns))]
        ax[0].plot(time_bsl, savgol_filter(average, filter_value, 2), '#253494')
        ax[0].axhline(y=0, ls='--', color='k')
        ax[0].set_xlabel('Time (sec)')
        ax[0].set_ylabel('DF/F')
    
        for i in range(len(peak_time_list)):
            peak_index = np.ravel(np.where(df_traces['Time'] <= peak_time_list[i]))[-1]
            start = np.ravel(np.where(df_traces['Time'] >= df_traces['Time'][peak_index]-0.04))[0]
            stop = np.ravel(np.where(df_traces['Time'] >= df_traces['Time'][peak_index]+0.06))[0]
            peak_trace = df_traces[df_traces.columns[col]][start:stop]
            PEAKS.append(np.array(peak_trace))
            ax[1].plot(df_traces['Time'][0:end_first_peak_idx], savgol_filter(peak_trace, filter_value, 2), '#1d91c0', alpha=0.2)
    
        avg_peak = np.mean(PEAKS, axis=0)
        PEAK_EP.append(avg_peak)
        ax[1].plot(df_traces['Time'][0:end_first_peak_idx], savgol_filter(avg_peak, filter_value, 2), '#1d91c0', lw=2)
        ax[1].axhline(y=0, ls='--', color='k')
        ax[1].set_title(f'{df_traces.columns[col]}')
        ax[1].set_ylabel('DF/F')
        ax[1].set_xlabel('Time(sec)')
        
        ax[2].plot(time_burst, savgol_filter(BURST[col], filter_value, 2), color = '#7fcdbb')
        ax[2].axhline(y=0, ls='--', color='k')
        ax[2].set_title(f'Ep{col+1}')
        ax[2].set_ylabel('DF/F')
        ax[2].set_xlabel('Time (sec)')
        
        # plt.figure()
        # plt.plot(df_traces['Time'], df_traces[col], 'g', alpha=0.2)
        # plt.plot(df_traces['Time'], savgol_filter(df_traces[col], filter_value, 2), 'g')
        # plt.title(f'{col}')
        # plt.ylabel('DF/F')
        # plt.xlabel('Time(sec)')
        
        # df_peaks = pd.concat((pd.DataFrame(PEAKS).transpose(),
        #                       pd.DataFrame(avg_peak, columns=['Average']),
        #                       pd.DataFrame(df_traces['Time'][0:end_first_peak_idx], columns=['Time'])), axis=1)
        
        # with pd.ExcelWriter('{}/{}_recovery_{}_LFDpeak_traces_2.5mMCa_button{}_dF_F0.xlsx'.format(Path, Linescan, freq, col)) as writer:
        #     df_peaks.to_excel(writer, index=False)
            
    # [plt.plot(TIME[i], savgol_filter(EPISODE[i], filter_value, 2), 'k') for i in range(len(EPISODE))]


plt.figure()
plt.plot(df_traces['Time'], df_traces['Avg'], 'g', alpha=0.2)
plt.plot(df_traces['Time'], savgol_filter(df_traces['Avg'], filter_value, 2), 'g')
[plt.plot(TIME[i], savgol_filter(EPISODE[i], filter_value, 2), 'k') for i in range(len(EPISODE))]
plt.title('Avg')
plt.ylabel('DF/F')
plt.xlabel('Time(sec)')


PEAKS_AVG = []
plt.figure()
for i in range(len(peak_time_list)):
    peak_index = np.ravel(np.where(df_traces['Time'] <= peak_time_list[i]))[-1]
    start = np.ravel(np.where(df_traces['Time'] >= df_traces['Time'][peak_index]-0.04))[0]
    stop = np.ravel(np.where(df_traces['Time'] >= df_traces['Time'][peak_index]+0.06))[0]
    PEAKS_AVG.append(df_traces['Avg'][start:stop])
    plt.plot(df_traces['Time'][0:end_first_peak_idx], savgol_filter(df_traces['Avg'][start:stop], filter_value,2), color='#1d91c0', alpha=0.2)
plt.plot(df_traces['Time'][0:end_first_peak_idx], np.mean(PEAKS_AVG, axis=0), '#1d91c0', lw=2)
plt.title('Mean of mean peaks')
plt.ylabel('DF/F')
plt.xlabel('Time(sec)')


df = pd.concat((pd.DataFrame(PEAK_EP).transpose(), pd.DataFrame(np.mean(PEAK_EP, axis=0), columns=['Average']),
                pd.DataFrame(df_traces['Time'][0:end_first_peak_idx], columns = ['Time'])), axis=1)

if Save == True:
    with pd.ExcelWriter('{}/{}_{}_avg_peak_traces_end_LFD_2.5mMCa_dF_F0.xlsx'.format(Path, Linescan, freq)) as writer:
        df.to_excel(writer, index=False)
        

