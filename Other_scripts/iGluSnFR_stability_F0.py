# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 17:24:54 2022

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
name_excel_file = 'iGluSnFR_Amp2_F0_averages'




files = sorted(os.listdir(Path))

INDEXES = []
F0 = []
INDEXES_TAU = []
TAU = []

for file in tqdm(range(len(files))):
    if freq in files[file]:
        if calcium in files[file]:
            if 'traces_converted.xlsx' in files[file]:
                name = files[file].rsplit('_',2)[0]
                data = pd.read_excel(f'{Path}/{files[file]}', sheet_name='F0')
                f0_avg = data['Average'].values
                INDEXES.append(name)
                F0.append(f0_avg)
            if 'Tau.xlsx' in files[file]:
                name = files[file].rsplit('_',2)[0]
                data = pd.read_excel(f'{Path}/{files[file]}')
                f0_avg = data['Average'].values
                INDEXES.append(name)
                F0.append(f0_avg)



data_cleaned = pd.read_excel(f'{Path_cleaned_data}/{file_cleaned_data}', sheet_name='20Hz_2,5mM')
INDEXES_CLEANED = []
A1_CLEANED, A2_CLEANED = [],[]
Psyn1_CLEANED, Psyn2_CLEANED = [],[]
PPR2_1 = []
F0_CLEANED = []

for i in range(len(INDEXES)):
    for j in range(len(data_cleaned['ID'])):
        if INDEXES[i] == data_cleaned['ID'][j]:
            INDEXES_CLEANED.append(INDEXES[i])
            A1_CLEANED.append(data_cleaned['AMP1'][j])
            A2_CLEANED.append(data_cleaned['AMP2'][j])
            Psyn1_CLEANED.append(1-(data_cleaned['%Fail1'][j]/100))
            Psyn2_CLEANED.append(1-(data_cleaned['%Fail2'][j]/100))
            PPR2_1.append(data_cleaned['PPR2/1'][j])
            F0_CLEANED.append(F0[i])

df = pd.concat((pd.DataFrame(INDEXES_CLEANED),
                pd.DataFrame(A1_CLEANED),
                pd.DataFrame(A2_CLEANED),
                pd.DataFrame(Psyn1_CLEANED),
                pd.DataFrame(Psyn2_CLEANED),
                pd.DataFrame(PPR2_1),
                pd.DataFrame(F0_CLEANED)), axis=1)
df.columns = ['ID', 'AMP1', 'AMP2', 'Psyn1', 'Psyn2', 'PPR2/1', 'F0_AVG']

if save_excel == True:
    with pd.ExcelWriter(f'{Path_cleaned_data}/{name_excel_file}.xlsx') as writer:
        df.to_excel(writer, sheet_name='20Hz_2.5mM', index=False)