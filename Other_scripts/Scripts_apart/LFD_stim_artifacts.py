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


Path = r'E:\AAVDJ.GluSnFR-S72A\iGluSnFR-S72A_LFD'
freq = '50Hz'
Linescan = 'linescan2'
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
### LOAD RECOVERY FILES #######
##### MAKE FILES UNION ########
###############################
    
files = sorted(os.listdir(Path))

RECOVERY1, RECOVERY2 = [],[]
for file in files:
    if freq in file:
        if 'dF_F0.xlsx' in file:
            if Linescan in file:
                if 'recovery1' in file:
                    recovery1 = pd.read_excel('{}/{}'.format(Path, file))
                    trace = recovery1.iloc[:,0]
                    time1 = recovery1['Time']
                    RECOVERY1.append(trace)
                        
                elif 'recovery2' in file:
                    recovery2 = pd.read_excel('{}/{}'.format(Path, file))
                    trace = recovery2.iloc[:,0]
                    time2 = recovery2['Time']
                    RECOVERY2.append(trace)

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


for col in df_traces.columns:
    if 'Ep' in col:
        plt.figure()
        plt.plot(df_traces['Time'], df_traces[col], 'k', alpha=0.3)
        plt.plot(df_traces['Time'], savgol_filter(df_traces[col], filter_value, 2), 'r')
        plt.title(f'{col}')
    [plt.plot(TIME[i], EPISODE[i], 'k') for i in range(len(EPISODE))]


plt.figure()
plt.plot(df_traces['Time'], df_traces['Avg'], alpha=0.5)
plt.plot(df_traces['Time'], savgol_filter(df_traces['Avg'], filter_value, 2), 'r')
[plt.plot(TIME[i], EPISODE[i], 'k') for i in range(len(EPISODE))]
plt.title('Avg')



if Save == True:
    with pd.ExcelWriter('{}/20210928_linescan1_recovery1AVG_{}_3pulses_2.5mMCa_dF_F0.xlsx'.format(Path, freq)) as writer:
        df1.to_excel(writer)
    
    with pd.ExcelWriter('{}/20210928_linescan1_recovery2AVG_{}_3pulses_2.5mMCa_dF_F0.xlsx'.format(Path, freq)) as writer:
        df2.to_excel(writer)