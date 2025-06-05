# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:21:21 2020

@author: Theo.ROSSI
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


#################################SETTINGS######################################
###############################################################################

Path = r'\\equipe2-nas1\Theo.ROSSI\GluSnFR\Virus_AAV.hSynap.SF-iGluSnFR_S72A\AAVDJ.GluSnFR-S72A'
Excel_files = 'AAVDJ.GluSnFR-S72A_STP_raw_profiles_20Hz'
Storage_file = 'AAVDJ.GluSnFR-S72A_20Hz_Ca15_Ca25_PF_amp_PEAK3-PEAK2.xlsx'


Dataframe = True
###############################################################################
###############################################################################


files = sorted(os.listdir('{}'.format(Path)))


LIST_RAW_VALUES = []
LIST_PPR3_2 = []

labels = ['[Ca2+]e = 1.5mM', '[Ca2+]e = 2.5mM']
colors = ['darkturquoise', 'limegreen']

for file in range(len(files)):

    if Excel_files in files[file]:
        data = pd.read_excel('{}/{}'.format(Path, files[file]), header=None, index=None)
        
        raw_values = data.iloc[:, 1:11].values
        LIST_RAW_VALUES.append(raw_values)
        
        
        for profil in range(raw_values.shape[0]):

            normalized_value = raw_values[profil, 2]/raw_values[profil, 1]
            LIST_PPR3_2.append(normalized_value)


fig, ax = plt.subplots(1)
mean = [np.mean(x) for x in [LIST_PPR3_2[0:7], LIST_PPR3_2[7:-1]]]
sem = [np.std(x)/np.sqrt(len(x)) for x in [LIST_PPR3_2[0:7], LIST_PPR3_2[7:-1]]]

for average, error, distribution, index in zip(mean, sem, [LIST_PPR3_2[0:7], LIST_PPR3_2[7:-1]], np.arange(len(labels))):
#    print ('------------------')
#    print(index,average,distribution)*
    
    width = 0.2
    step = 5.
    
    ax.bar(index, average, width=width, label=labels[index], color=colors[index], alpha=0.2)
    ax.errorbar(index, average, yerr=error, capsize=10, color='k')
    ax.legend(loc='best')
    
    x = np.linspace(index-(width/step), index+(width/step), len(distribution))
    ax.scatter(x, distribution, color=colors[index], alpha=0.5)
    ax.plot([-0.3,1.3], [1.,1.], 'k', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_title(Excel_files)
    ax.set_xlabel('Condition'), ax.set_ylabel('PPR3/2')


if Dataframe == True:
    df = pd.DataFrame([LIST_PPR3_2[0:7], LIST_PPR3_2[7:-1]])
    print(df)
    with pd.ExcelWriter('{}/{}'.format(Path, Storage_file)) as writer:
        df.to_excel(writer, header=False)
   