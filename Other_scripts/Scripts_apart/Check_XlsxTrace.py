# -*- coding: utf-8 -*-
"""
Created on Sun May  2 16:50:31 2021

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats,signal
from scipy.signal import savgol_filter

path = r'E:\AAVDJ.GluSnFR-S72A\AAVDJ.GluSnFR-S72A_Good_CaMg_ratio\Saturation_curve\Saturation_profiles'

files = sorted(os.listdir(path))

BIG_LIST = []
LIST25, LIST4 = [],[]
TRACES25, TIMES25 = [],[]
TRACES4, TIMES4 = [],[]
for file in range(len(files)):
    if 'dF_F0' in files[file]:
        if '3-50' in files[file]:
            BIG_LIST.append(files[file])
            # plt.figure()
            data = pd.read_excel('{}/{}'.format(path,files[file]))
            trace = data.iloc[:,0].values
            time = data.iloc[:,3].values
            
            if '2.5mM' in files[file]:
                LIST25.append(files[file])
                TRACES25.append(trace)
                TIMES25.append(time)
                # plt.plot(time,trace,'b')
            if '4mM' in files[file]:
                LIST4.append(files[file])
                TRACES4.append(trace)
                TIMES4.append(time)
                # plt.plot(time,trace,'r')
            # plt.title('{}'.format(files[file]))
                

RESAMPLED_TRACES25, X25 = [],[]
RESAMPLED_TRACES4, X4 = [],[]
for trace in range(len(TRACES25)):
    start = np.ravel(np.where(TIMES25[trace] >= 0))[0]
    end = np.ravel(np.where(TIMES25[trace] <= 1.2))[-1]
    print(len(TRACES25[trace]))
    
    NewTrace = TRACES25[trace][start:end]
    resampling = signal.resample(NewTrace, 849)
    RESAMPLED_TRACES25.append(resampling)
    
    start_average = np.ravel(np.where(TIMES25[0] >= 0))[0]
    end_average = np.ravel(np.where(TIMES25[0] <= 1.2))[-1]
    x_average = np.linspace(TIMES25[0][start_average], TIMES25[0][end_average], num=849, endpoint=True)
    X25.append(x_average)
    
    # plt.plot(x_average, resampling, 'b', alpha=0.2)

for trace in range(len(TRACES4)):
    start = np.ravel(np.where(TIMES4[trace] >= 0))[0]
    end = np.ravel(np.where(TIMES4[trace] <= 1.2))[-1]
    print(len(TRACES4[trace]))
    
    NewTrace = TRACES4[trace][start:end]
    resampling = signal.resample(NewTrace, 849)
    RESAMPLED_TRACES4.append(resampling)
    
    start_average = np.ravel(np.where(TIMES4[0] >= 0))[0]
    end_average = np.ravel(np.where(TIMES4[0] <= 1.2))[-1]
    x_average = np.linspace(TIMES4[0][start_average], TIMES4[0][end_average], num=849, endpoint=True)
    X4.append(x_average)
    
    # plt.plot(x_average, resampling, 'r', alpha=0.2)

average25 = np.mean(RESAMPLED_TRACES25, axis=0)
average4 = np.mean(RESAMPLED_TRACES4, axis=0)
plt.figure()
plt.plot(X25[0], savgol_filter(average25, 5, 2), 'b', label='Ca 2.5mM')
plt.plot(X4[0], savgol_filter(average4, 5, 2), 'r', label='Ca 4mM')
plt.legend()

name25 = LIST25[0].rsplit('.',1)[0].split('_')
name25 = name25[0]+'_'+name25[1]+'_'+name25[4]+'_'+name25[5]

name4 = LIST4[0].rsplit('.',1)[0].split('_')
name4 = name4[0]+'_'+name4[1]+'_'+name4[4]+'_'+name4[5]

plt.figure()
plt.plot(X25[0], savgol_filter(RESAMPLED_TRACES25[0], 9, 2), 'b', label=name25)
plt.plot(X4[5], savgol_filter(RESAMPLED_TRACES4[5], 9, 2), 'r', label=name4)
plt.legend(loc='upper left')