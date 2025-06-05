# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:48:54 2022

@author: Theo.ROSSI
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

Path = r'E:\AAVDJ.GluSnFR-S72A\Boutons_analysis'
calcium = '2.5mM'
Freq = '20Hz'
Saving_path = r'E:\AAVDJ.GluSnFR-S72A'

files = sorted(os.listdir(Path))

fig, ax = plt.subplots()
df_averages = pd.DataFrame(index=None, columns=None)
df_single_traces = pd.DataFrame(index=None, columns=None)
for file in tqdm(range(len(files))):
    if Freq in files[file]:
        if calcium in files[file]:
            if 'traces_converted.xlsx' in files[file]:
                data = pd.read_excel(f'{Path}/{files[file]}', sheet_name='Traces DF_F0')
                time = data['Time']
                avg = data['Average']
                
                name = files[file].split('_')
                name = name[0]+'_'+name[1]+'_'+name[5]
                
                ep = data.drop(['Average','Time'], axis=1)
                avg_std = np.std(ep, axis=1)
                
                df_averages[f'{name}_time'] = time
                df_averages[f'{name}_avg'] = avg
                df_averages[f'{name}_std'] = avg_std
            
                # for i in range(ep.shape[1]):
                #     df_single_traces[f'{name}_{i}'] = ep.iloc[:,i]
                
                # ax.plot(time, avg)

with pd.ExcelWriter(f'{Saving_path}\iGluSnFR_all_episode_traces_2.5mMCa_{Freq}.xlsx') as writer:
    df_averages.to_excel(writer)