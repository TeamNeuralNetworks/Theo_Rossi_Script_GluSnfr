# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:33:32 2022

@author: theo.rossi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy.stats as stats
from scipy.signal import savgol_filter
from tqdm import tqdm

Path = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Controls\iGluSnFR.S72A_forskoline'
Path_filtered_data = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Tidy_files'
file_filtered_data = 'GluSnFR_avg_clustering_variables_all_profiles_20Hz_forskoline_filtered_3sigma.xlsx'
calcium = '1.5mM'
freq = '20Hz'
sheet_peak = 'PEAK2'
name_saved_file = 'A2_no_fail_FSK_20Hz'
save = True


data_filtered = pd.read_excel(f'{Path_filtered_data}/{file_filtered_data}', sheet_name='Amps1&2_no_fail_1.5mMCa')
NAME_FILTERED = [data_filtered['ID'][i] for i in range(data_filtered.shape[0]) if '1.5mM' in data_filtered['ID'][i]]



files = sorted(os.listdir(Path))

NAME = []
TRACES = []
EP_PEAK = []

for file in tqdm(range(len(files))):
    if calcium in files[file]:
        if freq in files[file]:
            if 'traces_converted.xlsx' in files[file]:
                name = files[file].rsplit('_',2)[0]
                NAME.append(name)
                data_traces = pd.read_excel(f'{Path}/{files[file]}', sheet_name='Traces DF_F0')
                trials = data_traces.drop('Average', axis=1)
                TRACES.append(trials)
            elif 'data_bootstrap.xlsx' in files[file]:
                data_bootstrap = pd.read_excel(f'{Path}/{files[file]}', sheet_name=sheet_peak)
                ep_peak = data_bootstrap.iloc[:,:2]
                EP_PEAK.append(ep_peak)



AMP = []
for item in range(len(NAME)):
    for filtered_item in range(len(NAME_FILTERED)):
        if NAME[item] == NAME_FILTERED[filtered_item]:
            SUCCESS_TRACES = []
            TIME = []
            for i in range(len(TRACES[item].columns)):
                if i == TRACES[item].columns.tolist().index('Time'):
                    continue
                else:
                    if TRACES[item].columns[i] in EP_PEAK[item].iloc[i,0]:
                        if 'Success' in EP_PEAK[item].iloc[i,1]:
                            SUCCESS_TRACES.append(TRACES[item].iloc[:,i])
                            TIME.append(TRACES[item]['Time'])
                            # print(NAME[item])
            avg = savgol_filter(np.mean(SUCCESS_TRACES, axis=0), 9, 2)
            
            if sheet_peak == 'PEAK1':
                a = 0.498
                b = 0.51
            elif sheet_peak == 'PEAK2':
                a = 0.548
                b = 0.56
                
            start = np.ravel(np.where(TIME[0] <= a))[-1]
            stop = np.ravel(np.where(TIME[0] <= b))[-1]
            
            peak = avg[start:stop]
            amp = np.max(peak)
            AMP.append(amp)
            
            plt.figure()
            plt.plot(TIME[0], avg)
            plt.title(f'{NAME[item]}')
            plt.axhline(y=0, color='k', ls='--')
            plt.axhline(y=amp, color='r')

df = pd.concat((pd.DataFrame(NAME_FILTERED),
                pd.DataFrame(AMP)), axis=1)
df.columns = ['ID', 'AMP2']

if save == True:
    with pd.ExcelWriter(f'{Path_filtered_data}/{name_saved_file}.xlsx') as writer:
        df.to_excel(writer, index=False)