# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:21:47 2021

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import optimize



def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)


path = r'\\equipe2-nas1\Theo.ROSSI\NeuroFrance 2021\Figures'

files = sorted(os.listdir(path))

for file in range(len(files)):
    if 'bouton4_PC_plotprofile_green' in files[file]:
        baseline = pd.read_csv('{}/{}'.format(path,files[file]), sep='\t',header=1).iloc[:,0]
    if 'bouton4_PC_plotprofile_red.csv' in files[file]:
        signal = pd.read_csv('{}/{}'.format(path,files[file]), sep='\t',header=1).iloc[:,0]

distance_bsl, intensity_bsl = [],[]
distance_sig, intensity_sig = [],[]
for i in range(len(baseline)):
    basic_value = baseline[i]
    basic_signal = signal[i]
    print (basic_value)
    
    x_bsl = float(basic_value.split(',')[0])
    y_bsl = float(basic_value.split(',')[1])
    
    
    distance_bsl.append(x_bsl)
    intensity_bsl.append(y_bsl)
    
    x_sig = float(basic_signal.split(',')[0])
    y_sig = float(basic_signal.split(',')[1])
    
    distance_sig.append(x_sig)
    intensity_sig.append(y_sig)

leak_intensity_bsl = [intensity_bsl[i]-np.min(intensity_bsl) for i in range(len(intensity_bsl))]
leak_intensity_sig = [intensity_sig[i]-np.min(intensity_sig) for i in range(len(intensity_sig))]

norm_intensity_bsl = [leak_intensity_bsl[i]/np.max(leak_intensity_bsl) for i in range(len(intensity_bsl))]
norm_intensity_sig = [leak_intensity_sig[i]/np.max(leak_intensity_sig) for i in range(len(intensity_sig))]

# popt_bsl, _ = optimize.curve_fit(gaussian, distance_bsl, intensity_bsl)
# popt_sig, _ = optimize.curve_fit(gaussian, distance_sig, intensity_sig)

plt.plot(distance_bsl, norm_intensity_bsl,'k')
plt.plot(distance_sig, norm_intensity_sig, 'g')
# plt.plot(distance_bsl, gaussian(distance_bsl, *popt_bsl), 'k', lw=3)
# plt.plot(distance_sig, gaussian(distance_sig, *popt_sig), 'g', lw=3)