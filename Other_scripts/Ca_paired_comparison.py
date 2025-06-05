# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:22:57 2022

@author: theo.rossi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statannot import add_stat_annotation

Path = r"\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Tidy_files"

file = 'GluSnFR_avg_clustering_variables_all_profiles_1.5_4mM_Ca_paired_filtered_3sigma.xlsx'
file_all_Ca = 'GluSnFR_avg_clustering_variables_all_profiles_1.5vs2.5vs4mMCa_filtered_3sigma.xlsx'

freq = ['20Hz', '50Hz']

data_20Hz = pd.read_excel(f'{Path}/{file}', sheet_name=freq[0])
data_50Hz = pd.read_excel(f'{Path}/{file}', sheet_name=freq[1])
datasets = [data_20Hz, data_50Hz]


name = []

all_amps_15, all_amps_4 = [],[]
a1_15, a1_4 = [],[]
fail1_15, fail1_4 = [],[]
ppr_15_20Hz, ppr_4_20Hz = [],[]
ppr_15_50Hz, ppr_4_50Hz = [],[]

for dataset in range(len(datasets)):
    for i in range(datasets[dataset].shape[0]):
        if '1.5mM' in datasets[dataset].iloc[i,0]:
            a1_15.append(datasets[dataset].iloc[i,1])
            fail1_15.append(datasets[dataset]['%Fail1'][i])
            if dataset == 0:
                name.append(datasets[dataset].iloc[i,0])
                all_amps_15.append(datasets[dataset].iloc[i,2:11])
                ppr_15_20Hz.append(datasets[dataset].iloc[i,11])
            elif dataset == 1:
                ppr_15_50Hz.append(datasets[dataset].iloc[i,11])
            # df.loc[len(df.index)] = [datasets[dataset].iloc[i,0], datasets[dataset].iloc[i,1], i, datasets[dataset].iloc[i,11], i]
        elif '4mM' in datasets[dataset].iloc[i,0]:
            a1_4.append(datasets[dataset].iloc[i,1])
            fail1_4.append(datasets[dataset]['%Fail1'][i])
            if dataset == 0:
                all_amps_4.append(datasets[dataset].iloc[i,1:11])
                ppr_4_20Hz.append(datasets[dataset].iloc[i,11])
            elif dataset == 1:
                ppr_4_50Hz.append(datasets[dataset].iloc[i,11])
            # df['AMP1_4mM'].replace({df.iloc[i-df.shape[0],2]: datasets[dataset].iloc[i,1]}, inplace=True)
            # df['PPR_4mM'].replace({df.iloc[i-df.shape[0],4]: datasets[dataset].iloc[i,11]}, inplace=True)


df_a1 = pd.concat((pd.DataFrame(a1_15),
                   pd.DataFrame(a1_4)), axis=1)
df_a1.columns = ['A1_1.5mM', 'A1_4mM']

df_ppr = pd.concat((pd.DataFrame(ppr_15_20Hz),
                    pd.DataFrame(ppr_4_20Hz),
                    pd.DataFrame(ppr_15_50Hz),
                    pd.DataFrame(ppr_4_50Hz)), axis=1)
df_ppr.columns = ['PPR_15_20Hz','PPR_4_20Hz','PPR_15_50Hz','PPR_4_50Hz']

fig, ax =plt.subplots(1,2,tight_layout=True)
sns.boxplot(data=df_a1, ax=ax[0], showmeans=True)
sns.boxplot(data=[df_ppr['PPR_15_20Hz'],
                  df_ppr['PPR_4_20Hz'],
                  df_ppr['PPR_15_50Hz'].dropna(),
                  df_ppr['PPR_4_50Hz'].dropna()], ax=ax[1], showmeans=True)


[ax[0].plot([0,1], [a1_15[i], a1_4[i]], 'k', marker='o', alpha=0.2) for i in range(len(a1_15))]
[ax[1].plot([0,1], [df_ppr['PPR_15_20Hz'][i], df_ppr['PPR_4_20Hz'][i]], 'k', marker='o', alpha=0.2) for i in range(df_ppr['PPR_15_20Hz'].shape[0])]
[ax[1].plot([2,3], [df_ppr['PPR_15_50Hz'][i], df_ppr['PPR_4_50Hz'][i]], 'k', marker='o', alpha=0.2) for i in range(df_ppr['PPR_15_50Hz'].shape[0])]

ax[0].set_xticklabels(['1.5mM','4mM'], rotation=45, ha='right')
ax[0].set_ylabel('A1 (DF/F)')
ax[0].set_ylim(0)
ax[1].set_xticklabels(['PPR_1.5mM_20Hz','PPR_4mM_20Hz','PPR_1.5mM_50Hz','PPR_4mM_50Hz'], rotation=45, ha='right')
ax[1].axhline(y=1, color='r', ls='--')
ax[1].set_ylabel('PPR2/1')
ax[1].set_ylim(0)

add_stat_annotation(data=df_a1, ax=ax[0], box_pairs = [('A1_1.5mM', 'A1_4mM')], test = 'Wilcoxon', text_format='full')
add_stat_annotation(data=df_ppr, ax=ax[1], box_pairs = [('PPR_15_20Hz', 'PPR_4_20Hz')], test = 'Wilcoxon', text_format='full')
add_stat_annotation(data=df_ppr, ax=ax[1], box_pairs = [('PPR_15_50Hz', 'PPR_4_50Hz')], test = 'Wilcoxon', text_format='full')

##### A1 NO FAIL HISTOGRAM
##########################

data_no_fail_a1_15 = pd.read_excel(f'{Path}/{file_all_Ca}', sheet_name='Amp1_no_fail_1,5mMCa')
data_no_fail_a1_4 = pd.read_excel(f'{Path}/{file_all_Ca}', sheet_name='Amp1_no_fail_4mMCa')

mean_15 = data_no_fail_a1_15['AMP1'].mean()
mean_4 = data_no_fail_a1_4['AMP1'].mean()

fig_no_fail, ax_no_fail = plt.subplots()
sns.histplot(data=[list(data_no_fail_a1_15['AMP1']), list(data_no_fail_a1_4['AMP1'])], binwidth=0.05, palette=['grey','r'], ax=ax_no_fail, stat='probability')

##### A1 Psyn
##########################

Psyn_15 = [1-(fail1_15[i]/100) for i in range(len(fail1_15))]
Psyn_4 = [1-(fail1_4[i]/100) for i in range(len(fail1_4))]
df_Psyn = pd.concat((pd.DataFrame(Psyn_15, columns=['Psyn_1.5mM']),
                     pd.DataFrame(Psyn_4, columns=['Psyn_4mM'])), axis=1)

fig_Psyn, ax_Psyn = plt.subplots()
sns.boxplot(data=df_Psyn, ax=ax_Psyn, palette=['grey','r'], showmeans=True)
ax_Psyn.plot([0,1], [df_Psyn['Psyn_1.5mM'], df_Psyn['Psyn_4mM']], 'k', marker='o', alpha=0.2)
ax_Psyn.set_ylim(0,1.1)

add_stat_annotation(data=df_Psyn, ax=ax_Psyn, box_pairs = [('Psyn_1.5mM', 'Psyn_4mM')], test = 'Wilcoxon', text_format='full')
# sns.histplot(data=df_Psyn, binwidth=0.08, palette=['grey','r'], ax=ax_Psyn, stat='probability')

##### A1 N
##########################

data_no_fail_a1_15_pair = pd.read_excel(f'{Path}/{file}', sheet_name='Amp1_no_fail_1,5mMCa')
data_no_fail_a1_4_pair = pd.read_excel(f'{Path}/{file}', sheet_name='Amp1_no_fail_4mMCa')

name_no_fail_20Hz = [data_no_fail_a1_15_pair['ID'][i] for i in range(data_no_fail_a1_15_pair.shape[0]) if '20Hz' in data_no_fail_a1_15_pair['ID'][i]]
a1_no_fail_15mM_20Hz = [data_no_fail_a1_15_pair['AMP1'][i] for i in range(data_no_fail_a1_15_pair.shape[0]) if '20Hz' in data_no_fail_a1_15_pair['ID'][i]]
a1_no_fail_4mM_20Hz = [data_no_fail_a1_4_pair['AMP1'][i] for i in range(data_no_fail_a1_4_pair.shape[0]) if '20Hz' in data_no_fail_a1_4_pair['ID'][i]]

a2_no_fail_15mM_20Hz = [data_no_fail_a1_15_pair['AMP2'][i] for i in range(data_no_fail_a1_15_pair.shape[0]) if '20Hz' in data_no_fail_a1_15_pair['ID'][i]]

N = [data_no_fail_a1_4_pair['AMP1'][i]/data_no_fail_a1_15_pair['AMP1'][i] for i in range(data_no_fail_a1_4_pair.shape[0])]
mean_N = np.mean(N)

fig_N, ax_N = plt.subplots()
sns.histplot(data=N, binwidth=0.5, palette=['r'], ax=ax_N, stat='probability')
ax_N.set_xlim(0)

##### CUMULATIVE PLOT
##########################

fig_cumulative_N, ax_cumulative_N = plt.subplots()
x = np.arange(1,11)
FOLD_15mM = []
FOLD_4mM = []
for i in range(len(name)):
    for j in range(len(name_no_fail_20Hz)):
        if name[i] == name_no_fail_20Hz[j]:
            amps2_nofail_2_to_10 = all_amps_15[i].replace(to_replace = all_amps_15[i][0], value = a2_no_fail_15mM_20Hz[j])
            amps_no_fail_4mM_20Hz = all_amps_4[i].replace(to_replace = all_amps_4[i][0], value = a1_no_fail_4mM_20Hz[j])
            
            fold_15mM = np.cumsum(pd.Series(1., index=['AMP1']).append(amps2_nofail_2_to_10/a1_no_fail_15mM_20Hz[j]))
            fold_4mM = np.cumsum(all_amps_4[i]/a1_no_fail_15mM_20Hz[j]).astype(float)
            
            FOLD_15mM.append(fold_15mM)
            FOLD_4mM.append(fold_4mM)
            
            ax_cumulative_N.plot(x, fold_15mM, 'k', alpha=0.2)
            ax_cumulative_N.plot(x, fold_4mM, 'r', alpha=0.2)

fold_15mM_mean = np.mean(FOLD_15mM, axis=0)
fold_15mM_sem = stats.sem(FOLD_15mM, axis=0)

fold_4mM_mean = np.mean(FOLD_4mM, axis=0)
fold_4mM_sem = stats.sem(FOLD_4mM, axis=0)

ax_cumulative_N.plot(x, fold_15mM_mean, 'k', marker='o')
ax_cumulative_N.fill_between(x, fold_15mM_mean+fold_15mM_sem, fold_15mM_mean-fold_15mM_sem, color='k', alpha=0.4)

ax_cumulative_N.plot(x, fold_4mM_mean, 'r', marker='o')
ax_cumulative_N.fill_between(x, fold_4mM_mean+fold_4mM_sem, fold_4mM_mean-fold_4mM_sem, color='r', alpha=0.4)

ax_cumulative_N.set_ylabel('Cumulative N')
ax_cumulative_N.set_xlabel('Pulse number')
ax_cumulative_N.set_ylim(0)
ax_cumulative_N.set_title('20Hz')




profiles_20Hz_15mM = [data_20Hz.iloc[i,1:11].astype(float)/data_20Hz.iloc[i,1] for i in range(data_20Hz.shape[0]) if '1.5mM' in data_20Hz.iloc[i,0]]
profiles_20Hz_4mM = [data_20Hz.iloc[i,1:11].astype(float)/data_20Hz.iloc[i,1] for i in range(data_20Hz.shape[0]) if '4mM' in data_20Hz.iloc[i,0]]
profiles_50Hz_15mM = [data_50Hz.iloc[i,1:11].astype(float)/data_50Hz.iloc[i,1] for i in range(data_50Hz.shape[0]) if '1.5mM' in data_50Hz.iloc[i,0]]
profiles_50Hz_4mM = [data_50Hz.iloc[i,1:11].astype(float)/data_50Hz.iloc[i,1] for i in range(data_50Hz.shape[0]) if '4mM' in data_50Hz.iloc[i,0]]

mean_profiles_20Hz_15mM = np.mean(profiles_20Hz_15mM, axis=0)
mean_profiles_20Hz_4mM = np.mean(profiles_20Hz_4mM, axis=0)
mean_profiles_50Hz_15mM = np.mean(profiles_50Hz_15mM, axis=0)
mean_profiles_50Hz_4mM = np.mean(profiles_50Hz_4mM, axis=0)

sem_profiles_20Hz_15mM = stats.sem(profiles_20Hz_15mM)
sem_profiles_20Hz_4mM = stats.sem(profiles_20Hz_4mM)
sem_profiles_50Hz_15mM = stats.sem(profiles_50Hz_15mM)
sem_profiles_50Hz_4mM = stats.sem(profiles_50Hz_4mM)


fig_norm_profile, ax_norm_profile = plt.subplots(1,2, tight_layout=True)
[ax_norm_profile[0].plot(x, profiles_20Hz_15mM[i], 'k', lw=2, alpha=0.1) for i in range(len(profiles_20Hz_15mM))]
[ax_norm_profile[0].plot(x, profiles_20Hz_4mM[i], 'r', lw=2, alpha=0.1) for i in range(len(profiles_20Hz_4mM))]
[ax_norm_profile[1].plot(x, profiles_50Hz_15mM[i], 'k', lw=2, alpha=0.1) for i in range(len(profiles_50Hz_15mM))]
[ax_norm_profile[1].plot(x, profiles_50Hz_4mM[i], 'r', lw=2, alpha=0.1) for i in range(len(profiles_50Hz_4mM))]

ax_norm_profile[0].plot(x, mean_profiles_20Hz_15mM, 'k', lw=2, marker='o')
ax_norm_profile[0].plot(x, mean_profiles_20Hz_4mM, 'r', lw=2, marker='o')
ax_norm_profile[1].plot(x, mean_profiles_50Hz_15mM, 'k', lw=2, marker='o')
ax_norm_profile[1].plot(x, mean_profiles_50Hz_4mM, 'r', lw=2, marker='o')

ax_norm_profile[0].fill_between(x, mean_profiles_20Hz_15mM+sem_profiles_20Hz_15mM, mean_profiles_20Hz_15mM-sem_profiles_20Hz_15mM, 'k', alpha=0.5)
ax_norm_profile[0].fill_between(x, mean_profiles_20Hz_4mM+sem_profiles_20Hz_4mM, mean_profiles_20Hz_4mM-sem_profiles_20Hz_4mM, 'r', alpha=0.5)
ax_norm_profile[1].fill_between(x,  mean_profiles_50Hz_15mM+sem_profiles_50Hz_15mM, mean_profiles_50Hz_15mM-sem_profiles_50Hz_15mM, 'k', alpha=0.5)
ax_norm_profile[1].fill_between(x,  mean_profiles_50Hz_4mM+sem_profiles_50Hz_4mM, mean_profiles_50Hz_4mM-sem_profiles_50Hz_4mM, 'r', alpha=0.5)


ax_norm_profile[0].set_ylim(0,6)
ax_norm_profile[1].set_ylim(0,6)