# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:21:51 2020

@author: Theo.ROSSI
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#################################SETTINGS######################################
###############################################################################

Path = r'\\equipe2-nas1\Theo.ROSSI\GluSnFR\Virus_AAV.hSynap.SF-iGluSnFR_S72A\AAVDJ.GluSnFR-S72A'
Excel_files = '20Hz_Ca15_PF_normalized_amp_1'
Storage_file = 'AAVDJ.GluSnFR-S72A_STP_normalized_profiles_20Hz_Ca15_PF.xlsx'

Dataframe = False
'''Turn True to import button's data in the excel file'''

###############################################################################
###############################################################################

LIST_LINESCANS = []
LIST_DATA = []
LIST_VALUES = []
LIST_PROFILS = []

files = sorted(os.listdir('{}'.format(Path)))

for folder in range(len(files)):
    if '2019' in files[folder]:
        path2 = os.listdir('{}/{}'.format(Path, files[folder]))
        
        for file in range(len(path2)):
            if '100Hz' in path2[file]:
                continue
              
            if Excel_files in path2[file]:
                linescans = '{}'.format(path2[file])
                LIST_LINESCANS.append(linescans)
                
                data = pd.read_excel('{}/{}/{}'.format(Path, files[folder], path2[file]), header=None, index=None)
                LIST_DATA.append(data)
                
                values = data.iloc[:, 1:11].values
                LIST_VALUES.append(values)
                
                for profil in values:
                    LIST_PROFILS.append(profil)

                   
mean = np.nanmean(LIST_PROFILS, axis=0)                    
              
plt.figure(figsize=(8,4))
x = np.arange(0,10, step=1)

font = {'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 12,}

for i in range(len(LIST_PROFILS)):

    plt.plot(LIST_PROFILS[i], 'grey', linewidth=2, alpha=0.4), plt.grid(axis='y', linestyle='--')
    plt.scatter(x, LIST_PROFILS[i], c='turquoise')
    plt.plot(mean, 'teal', linewidth=2, alpha=0.2)
    plt.scatter(x, mean, c='teal', alpha=0.2)
    plt.xticks(x), plt.title('{}; Npf = {}, Nbt = {}'.format(Excel_files, len(LIST_LINESCANS), len(LIST_PROFILS)), fontdict=font), plt.xlabel('#STIM'), plt.ylabel('PEAKn/PEAK1')
    plt.plot([0,9], [1,1], 'k', linestyle='--', linewidth=1)
    plt.ylim(0, 5.0)


if Dataframe == True:
    df = pd.DataFrame(LIST_PROFILS)
    print(df)
    with pd.ExcelWriter('{}/{}'.format(Path, Storage_file)) as writer:
        df.to_excel(writer, header=False)
    
    