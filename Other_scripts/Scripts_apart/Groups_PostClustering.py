# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 22:52:39 2022

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
from statannot import add_stat_annotation
import scipy.stats as stats
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from itertools import combinations


File = r'\\equipe2-nas1\Theo.ROSSI\Paper_thesis\Figure5_PCvsMLI\MLI_HCPC_20Hz_4clusters.xlsx'
File_profils = r'E:\AAVDJ.GluSnFR-S72A\GluSnFR_avg_clustering_variables_all_profiles_unsupervised_new.xlsx'
File_all_avg_traces = r'\\equipe2-nas1\Theo.ROSSI\Paper_thesis\Figure5_PCvsMLI\Avg_traces_MLI_20Hz.xlsx'

freq = '20Hz'

def Colors(num):
    clr = []
    cmap = cm.viridis(np.linspace(0.1,0.85,num))
    for c in range(num):
        rgba = cmap[c]
        clr.append(colors.rgb2hex(rgba))
    return clr

data = pd.read_excel(File)
clusters = data.iloc[:,-1]
labels = Counter(clusters).keys()
clr = Colors(len(labels))

data_profils = pd.read_excel(File)
ppr_idx1 = list(data_profils.columns).index('PPR2/1')
ppr_idx2 = list(data_profils.columns).index('PPR10/1')
ppr = pd.concat((pd.DataFrame(np.ones(data_profils.shape[0])), data_profils.iloc[:,ppr_idx1:ppr_idx2+1]), axis=1)


if '20Hz' in File_all_avg_traces:
    new_time = np.linspace(0., 0.62, num=620, endpoint=True)
if '50Hz' in File_all_avg_traces:
    new_time = np.linspace(0., 0.49, num=490, endpoint=True)

data_all_avg_traces = pd.read_excel(File_all_avg_traces).transpose()
AVG_TRACES, TIME_TRACES = [],[]
for idx in range(data_all_avg_traces.shape[0]):
    if 'time' in data_all_avg_traces.index[idx]:
        if '20190801' in data_all_avg_traces.index[idx]:
            start = np.ravel(np.where(data_all_avg_traces.iloc[idx,:] >= 0.9))[0]
            end = np.ravel(np.where(data_all_avg_traces.iloc[idx,:] <= 1.7))[-1]
        else:
            if '20Hz' in File_all_avg_traces:
                start = np.ravel(np.where(data_all_avg_traces.iloc[idx,:] >= 0.4))[0] 
                end = np.ravel(np.where(data_all_avg_traces.iloc[idx,:] <= 1.2))[-1] 
            
            elif '50Hz' in File_all_avg_traces:
                start = np.ravel(np.where(data_all_avg_traces.iloc[idx,:] >= 0.4))[0]
                end = np.ravel(np.where(data_all_avg_traces.iloc[idx,:] <= 0.9))[-1]
        time = np.array(data_all_avg_traces.iloc[idx,start:end])
        time = time-time[0]
        TIME_TRACES.append(time)
        
    elif 'avg' in data_all_avg_traces.index[idx]:
        trace = data_all_avg_traces.iloc[idx,start:end]
        leak = np.mean(trace[0:100])
        trace = trace - leak
        f = interp1d(time, trace)
        y = f(new_time)
        AVG_TRACES.append(y)
        
        # plt.figure()
        # plt.plot(time, savgol_filter(trace,9,2))
        # plt.title(f'{data_all_avg_traces.index[idx]}')
    

plt.figure()
mean_traces = savgol_filter(np.mean(AVG_TRACES, axis=0), 9, 2)
std_traces = np.std(AVG_TRACES, axis=0)
plt.plot(new_time, mean_traces, 'r')
plt.fill_between(new_time, mean_traces+std_traces, mean_traces-std_traces, color='r', alpha=0.2)


fig, ax = plt.subplots(1,7, figsize=(14, 4), tight_layout=True)
x = np.arange(1, ppr.shape[1]+1, 1)

df_amp1 = pd.DataFrame(index=None, columns=None)
df_amp2 = pd.DataFrame(index=None, columns=None)
df_ppr2_1 = pd.DataFrame(index=None, columns=None)
df_fail1 = pd.DataFrame(index=None, columns=None)
df_fail2 = pd.DataFrame(index=None, columns=None)

for i in range(len(labels)):
    amp1 = [data['AMP1'][j] for j in range(data.shape[0]) if clusters[j] == i]
    amp2 = [data['AMP2'][j] for j in range(data.shape[0]) if clusters[j] == i]
    ppr2_1 = [data['PPR2/1'][j] for j in range(data.shape[0]) if clusters[j] == i]
    fail1 = [data['%Fail1'][j] for j in range(data.shape[0]) if clusters[j] == i]
    fail2 = [data['%Fail2'][j] for j in range(data.shape[0]) if clusters[j] == i]
    
    df_amp1[f'C{i}'] = pd.Series(amp1)
    df_amp2[f'C{i}'] = pd.Series(amp2)
    df_ppr2_1[f'C{i}'] = pd.Series(ppr2_1)
    df_fail1[f'C{i}'] = pd.Series(fail1)
    df_fail2[f'C{i}'] = pd.Series(fail2)
    
    episodes = [ppr.iloc[j,:] for j in range(ppr.shape[0]) if clusters[j] == i]
    mean = np.mean(episodes, axis=0)
    std = np.std(episodes, axis=0)
    # std = stats.sem(episodes, axis=0)
    
    group_avg_traces = [savgol_filter(AVG_TRACES[j], 9, 2) for j in range(len(AVG_TRACES)) if clusters[j] == i]
    mean_group_avg = np.mean(group_avg_traces, axis=0)
    std_group_avg = np.std(group_avg_traces, axis=0)
    

    ax[0].plot(x, mean, color=clr[i], marker='o')
    ax[0].fill_between(x, mean+std, mean-std, color=clr[i], alpha=0.3)
    
    ax[1].plot(new_time, mean_group_avg, color=clr[i])
    ax[1].fill_between(new_time, mean_group_avg+std_group_avg, mean_group_avg-std_group_avg, color=clr[i], alpha=0.3)
    
    # sns.histplot(data=amp1, bins=10, binwidth=0.05, ax=ax[2], color=clr[i], stat='density', kde=True)
    
    # normality_amp1 = stats.shapiro(amp1)
    # normality_ppr = stats.shapiro(ppr2_1)
    # normality_fail = stats.shapiro(fail1)
    # print(f'NORMALITY P-VALUE CLUSTER{i}')
    # print('----------------------------')
    # print(f'Amp1: {normality_amp1[1]}')
    # print(f'PPR: {normality_ppr[1]}')
    # print(f'Fail: {normality_fail[1]}')
    # print('----------------------------')
    
    # homogeneity_amp1 = stats.levene(amp1)
    # homogeneity_ppr = stats.shapiro(ppr2_1)
    # homogeneity_fail = stats.shapiro(fail1)
    # print('NORMALITY P-VALUE')
    # print(f'Amp1: {normality_amp1[1]}')
    # print(f'PPR: {normality_ppr[1]}')
    # print(f'Fail: {normality_fail1[1]}')
    
    
ax[0].set_title('Profiles')
ax[0].set_ylabel('An/A1')
ax[0].set_xlabel('Number stim')

fig_pie, ax_pie = plt.subplots()
proportions = [len(df_amp1.iloc[:,i].dropna()) for i in range(df_amp1.shape[1])]
profiles = [col for col in df_amp1.columns]
plt.pie(proportions, labels=profiles, colors=clr, startangle=90, autopct='%.1f%%')

pairs = list(combinations(df_ppr2_1.columns,2))
stat_test = 'Mann-Whitney'

#########################
#### TRACES FIGURES #####
#########################

### VIOLIN AMP1+AMP2 ####

sns.violinplot(data=df_amp1, ax=ax[2], palette=clr)
add_stat_annotation(ax[2], data=df_amp1, box_pairs=pairs, test=stat_test, text_format='full', verbose=2)
ax[2].set_title('Amp peak1')
ax[2].set_ylabel('A1 (DF/F)')
ax[2].set_xticklabels(df_amp1.columns)

sns.violinplot(data=df_amp2, ax=ax[3], palette=clr)
add_stat_annotation(ax[3], data=df_amp2, box_pairs=pairs, test=stat_test, text_format='full', verbose=2)
ax[3].set_title('Amp peak2')
ax[3].set_ylabel('A2 (DF/F)')
ax[3].set_xticklabels(df_amp2.columns)
    
##### VIOLIN PPR2/1 ####

sns.violinplot(data=df_ppr2_1, ax=ax[4], palette=clr)
add_stat_annotation(ax[4], data=df_ppr2_1, box_pairs=pairs, test=stat_test, text_format='full', verbose=2)
ax[4].set_title('PPR2/1')
ax[4].set_ylabel('A2/A1')
ax[4].set_xticklabels(df_ppr2_1.columns)

## VIOLIN FAIL1+FAIL2 ##

sns.violinplot(data=df_fail1, ax=ax[5], palette=clr)
add_stat_annotation(ax[5], data=df_fail1, box_pairs=pairs, test=stat_test, text_format='full', verbose=2)
ax[5].set_title('Failure peak1')
ax[5].set_ylabel('Failure A1 (%)')
ax[5].set_xticklabels(df_fail1.columns)

sns.violinplot(data=df_fail2, ax=ax[6], palette=clr)
add_stat_annotation(ax[6], data=df_fail2, box_pairs=pairs, test=stat_test, text_format='full', verbose=2)
ax[6].set_title('Failure peak2')
ax[6].set_ylabel('Failure A2 (%)')
ax[6].set_xticklabels(df_fail2.columns)