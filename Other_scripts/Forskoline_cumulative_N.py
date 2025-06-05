# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 22:14:27 2022

@author: theo.rossi
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats

Path = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Tidy_files'
file = 'GluSnFR_avg_clustering_variables_all_profiles_20Hz_forskoline_filtered_3sigma.xlsx'


data = pd.read_excel(f'{Path}/{file}', sheet_name='Amps1&2_no_fail_1.5mMCa')

BEFORE, AFTER = [],[]
A1_BEFORE, A1_AFTER = [],[]
for i in range(data.shape[0]):
    if 'before' in data['ID'][i]:
        BEFORE.append(data.iloc[i,1:].astype(float))
        A1_BEFORE.append(data.iloc[i,1])
    elif 'after' in data['ID'][i]:
        AFTER.append(data.iloc[i,1:].astype(float))
        A1_AFTER.append(data.iloc[i,1])

fold_before = [np.cumsum(BEFORE[i]/BEFORE[i][0]) for i in range(len(BEFORE))]
fold_after = [np.cumsum(AFTER[i]/BEFORE[i][0]) for i in range(len(AFTER))]

N = [fold_after[i][0]/fold_before[i][0] for i in range(len(fold_before))]

mean_before = np.mean(fold_before, axis=0)
mean_after = np.mean(fold_after, axis=0)

before_sem = stats.sem(fold_before)
after_sem = stats.sem(fold_after)

x = np.arange(1,11)

plt.figure()
[plt.plot(x, fold_before[i], 'k', alpha=0.2) for i in range(len(fold_before))]
[plt.plot(x, fold_after[i], 'darkorange', alpha=0.2) for i in range(len(fold_after))]
plt.fill_between(x, mean_before+before_sem, mean_before-before_sem, color='k', alpha=0.5)
plt.fill_between(x, mean_after+after_sem, mean_after-after_sem, color='darkorange', alpha=0.5)
plt.plot(x, mean_before, 'k', marker='o', label='1.5mMCa')
plt.plot(x, mean_after, 'darkorange', marker='o', label='1.5mMCa + FSK')
plt.xlabel('Pulse number')
plt.ylabel('Cumulative N')
plt.ylim(0,40)
plt.legend()





fig, ax = plt.subplots()
sns.histplot(data=N, binwidth=0.2, ax=ax, stat='probability')
ax.set_xlabel('A1 N')
ax.set_xlim(0,3)




fig_hist_amp, ax_hist_amp = plt.subplots()
sns.histplot(data=[A1_BEFORE, A1_AFTER], binwidth=0.08, ax=ax_hist_amp, stat='probability')
ax.set_xlabel('A1 (dF/F)')



fold_before_last_pulse = [fold_before[i][-1] for i in range(len(fold_before))]
fold_after_last_pulse = [fold_after[i][-1] for i in range(len(fold_after))]
fig_box_fold_last_pulse, ax_box_fold_last_amp = plt.subplots()
sns.boxplot(data=[fold_before_last_pulse, fold_after_last_pulse], ax=ax_box_fold_last_amp, showmeans=True)
ax_box_fold_last_amp.set_ylim(0)
