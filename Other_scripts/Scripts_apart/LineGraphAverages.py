# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:58:50 2020

@author: Theo.ROSSI
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#################################SETTINGS######################################
###############################################################################

Path = r'\\equipe2-nas1\Theo.ROSSI\GluSnFR\Virus_AAV.hSynap.SF-iGluSnFR_S72A\AAVDJ.GluSnFR-S72A'
Excel_files = 'normalized_profiles_20Hz'

###############################################################################
###############################################################################


files = sorted(os.listdir('{}'.format(Path)))

LIST_VALUES = []

labels = ['[Ca2+]e = 1.5mM', '[Ca2+]e = 2.5mM']
colors = ['darkturquoise', 'limegreen']


for file in range(len(files)):

    if Excel_files in files[file]:
        data = pd.read_excel('{}/{}'.format(Path, files[file]), header=None, index=None)
        
        values = data.iloc[:, 1:11].values
        LIST_VALUES.append(values)


average = [np.nanmean(LIST_VALUES[condition], axis=0) for condition in range(len(LIST_VALUES))]
sem = [np.std(LIST_VALUES[condition], axis=0)/np.sqrt(len(LIST_VALUES[condition])) for condition in range(len(LIST_VALUES))]


fig, ax = plt.subplots(1)

for mean, error, index in zip(average, sem, np.arange(len(labels))):
    
    ax.fill_between(range(len(mean)), mean-error, mean+error, color=colors[index], alpha=0.1)
    ax.plot(mean, label=labels[index], color=colors[index])
    ax.scatter(range(len(mean)), mean, color=colors[index])
    
    ax.legend(loc='best'), ax.grid(axis='y', linestyle='-.')
    
    ax.plot([0.,9.], [1.,1.], 'k', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_title(Excel_files)
    ax.set_yticks(np.arange(1., 2.7, 0.5))
    ax.set_xlabel('#STIM'), ax.set_ylabel('PEAKn/PEAK1')