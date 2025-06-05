# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:47:42 2022

@author: theo.rossi
"""

import pandas as pd
import os
from tqdm import tqdm

Path = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Boutons_analysis'
Path_cleaned_data = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Tidy_files'
file_cleaned_data = 'GluSnFR_avg_clustering_variables_all_profiles_1.5vs2.5vs4mMCa_filtered_3sigma.xlsx'
freq = '20Hz'
calcium = '2.5mMCa'
save_excel = True
name_excel_file = 'iGluSnFR_Tau_averages'




files = sorted(os.listdir(Path))

INDEXES_20Hz, INDEXES_50Hz = [],[]
TAU_20Hz, TAU_50Hz = [],[]

for file in tqdm(range(len(files))):
    if calcium in files[file]:
        if 'Tau.xlsx' in files[file]:
            name = files[file].rsplit('_',3)[0]
            data = pd.read_excel(f'{Path}/{files[file]}')
            tau = data.iloc[0,1:].tolist()
            if '20Hz' in files[file]:
                INDEXES_20Hz.append(name)
                TAU_20Hz.append(tau)
            elif '50Hz' in files[file]:
                INDEXES_50Hz.append(name)
                TAU_50Hz.append(tau)


IDX = [INDEXES_20Hz, INDEXES_50Hz]
TAUS = [TAU_20Hz, TAU_50Hz]

data_cleaned_20Hz = pd.read_excel(f'{Path_cleaned_data}/{file_cleaned_data}', sheet_name='20Hz_2,5mM')
data_cleaned_50Hz = pd.read_excel(f'{Path_cleaned_data}/{file_cleaned_data}', sheet_name='50Hz_2,5mM')
df_cleaned = [data_cleaned_20Hz, data_cleaned_50Hz]

DF = []
for index in range(len(IDX)):
    INDEXES_CLEANED = []
    TAU_CLEANED = []
    for i in range(len(IDX[index])):
        for frame in df_cleaned:
            for j in range(len(frame['ID'])):
                if IDX[index][i] == frame['ID'][j]:
                    INDEXES_CLEANED.append(IDX[index][i])
                    TAU_CLEANED.append(TAUS[index][i])


    df = pd.concat((pd.DataFrame(INDEXES_CLEANED),
                    pd.DataFrame(TAU_CLEANED)), axis=1)
    
    tau_names = [f'TAU{i+1}' for i in range(len(TAU_CLEANED[0]))]
    tau_names.insert(0, 'ID')
    df.columns = tau_names
    DF.append(df)
 
    
with pd.ExcelWriter(f'{Path_cleaned_data}\{name_excel_file}.xlsx') as writer:
    DF[0].to_excel(writer, sheet_name='20Hz_2.5mM', index=False)
    DF[1].to_excel(writer, sheet_name='50Hz_2.5mM', index=False)