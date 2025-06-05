# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:02:43 2022

@author: theo.rossi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import scipy.stats as stats
import seaborn as sns


file = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Tidy_files\GluSnFR_avg_clustering_variables_all_profiles_1.5_4mM_Ca_paired_filtered_3sigma.xlsx'
freq = '20Hz'

data = pd.read_excel(f'{file}', sheet_name=freq)

a1_15mM = []
a1_25mM = []
a1_4mM = []
ppr_15mM = []
ppr_25mM = []
ppr_4mM = []

for idx in range(data.shape[0]):
    bouton = data.iloc[idx,:]
    a1 = bouton['AMP1']
    ppr = bouton['PPR2/1']
    
    if '1.5mM' in data.iloc[idx,0]: 
        a1_15mM.append(a1)
        ppr_15mM.append(ppr)
    elif '2.5mM' in data.iloc[idx,0]:
        a1_15mM.append(a1)
        ppr_15mM.append(ppr)
    elif '4mM' in data.iloc[idx,0]:
        a1_4mM.append(a1)
        ppr_4mM.append(ppr)
        

df = pd.concat((pd.DataFrame(a1_15mM),
                pd.DataFrame(a1_4mM),
                pd.DataFrame(ppr_15mM),
                pd.DataFrame(ppr_4mM)), axis=1)
df.columns = ['A1_1.5mM', 'A1_4mM', 'PPR_1.5mM', 'PPR_4mM',]


fig, ax = plt.subplots(1,2, tight_layout=True)
sns.boxplot(data=df.iloc[:,:2], ax=ax[0], showmeans=True)
sns.boxplot(data=df.iloc[:,2:], ax=ax[1], showmeans=True)
ax[0].set_ylim(0,3)
ax[0].set_ylabel('DF/F')
ax[1].set_ylim(0,4)
ax[1].set_ylabel('PPR')
ax[1].axhline(y=1., color='r', ls='--')



###### STATISTICS
#################

s_a1_15, p_a1_15 = stats.shapiro(a1_15mM)
s_a1_4, p_a1_4 = stats.shapiro(a1_4mM)

w_a1, p_a1 = stats.wilcoxon(a1_15mM, a1_4mM)

s_ppr_15, p_ppr_15 = stats.shapiro(ppr_15mM)
s_ppr_4, p_ppr_4 = stats.shapiro(ppr_4mM)

# t_ppr, p_ppr = stats.ttest_rel(ppr_15mM, ppr_4mM)
w_ppr, p_ppr = stats.wilcoxon(ppr_15mM, ppr_4mM)