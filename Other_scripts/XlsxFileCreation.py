# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:27:27 2021

@author: Theo.ROSSI
"""

import matplotlib.pyplot as plt
import pandas as pd
import os


Path = r'E:\AAVDJ.GluSnFR-S72A\Boutons_analysis'

freq = '50Hz'
calcium = '4mM'
save = True

files = sorted(os.listdir(Path))


FILES = []
ID = []
AVERAGE_TOTAL, TIME_TOTAL, TIME_TOTAL_LENGTH = [],[],[]
AVERAGE, TIME = [],[]
AMP_TOTAL, PPR_TOTAL = [],[]
ALL_FAIL = []

TAU_FILE = []
TAU1 = []


for file in range(len(files)):
    if freq in files[file]:
        if calcium in files[file]:
            if 'ROI.tif' in files[file]:
                print(files[file])
                
            if 'Amp.xlsx' in files[file]:
                data = pd.read_excel(f'{Path}/{files[file]}')
                FILES.append(files[file])
                ind = files[file].rsplit('_',3)[0]
                ID.append(ind)

                amp = data.iloc[0,1:].values
                AMP_TOTAL.append(amp)
                ppr = [amp[i]/amp[0] for i in range(amp.shape[0])]
                PPR_TOTAL.append(ppr)
                
            if 'traces_converted.xlsx' in files[file]:
                data = pd.read_excel(f'{Path}/{files[file]}', sheet_name='Traces DF_F0')
                for col in range(len(data.columns)):
                    if 'Average'==data.columns[col]:
                        average = data.iloc[:,col].values
                        AVERAGE_TOTAL.append(average)

                    elif 'Time'==data.columns[col]:
                        time = data.iloc[:,col].values
                        TIME_TOTAL.append(time)
                        TIME_TOTAL_LENGTH.append(len(time))
            
            if 'data_bootstrap.xlsx' in files[file]:
                boot = [pd.read_excel(f'{Path}/{files[file]}', sheet_name=f'PEAK{i+1}') for i in range(3)]
                
                FAILURES = []
                for j in range(len(boot)):
                    for i in range(len(boot[j].columns)):
                        if 'PercFail' == boot[j].columns[i]:
                            failure = boot[j].iloc[0,i]
                            FAILURES.append(failure)
                ALL_FAIL.append(FAILURES)
            
            if 'Tau.xlsx' in files[file]:
                tau = pd.read_excel(f'{Path}/{files[file]}').iloc[0,1]*1000
                TAU1.append(tau)

            
df = pd.concat((pd.DataFrame(ID), pd.DataFrame(AMP_TOTAL),
                pd.DataFrame([PPR_TOTAL[i][1:] for i in range(len(PPR_TOTAL))]),
                pd.DataFrame([ALL_FAIL[i][0] for i in range(len(ALL_FAIL))]),
                pd.DataFrame([ALL_FAIL[i][1] for i in range(len(ALL_FAIL))]),
                pd.DataFrame([ALL_FAIL[i][2] for i in range(len(ALL_FAIL))]),
                pd.DataFrame(TAU1)), axis=1)

df.columns = ['ID','AMP1','AMP2','AMP3','AMP4','AMP5','AMP6','AMP7','AMP8','AMP9','AMP10','PPR2/1','PPR3/1','PPR4/1','PPR5/1','PPR6/1','PPR7/1','PPR8/1','PPR9/1','PPR10/1','%Fail1','%Fail2','%Fail3','Tau1']

if save == True:
    with pd.ExcelWriter(f'E:\AAVDJ.GluSnFR-S72A\GluSnFR_avg_clustering_variables_all_profiles_{freq}_{calcium}_unsupervised.xlsx') as writer:
        df.to_excel(writer,index=False)