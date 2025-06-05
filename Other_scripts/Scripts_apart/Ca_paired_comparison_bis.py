# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 19:05:19 2021

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm, colors
import dabest
import numpy as np
import pandas as pd
import os
from scipy.signal import resample, savgol_filter
import scipy.stats as stats
import seaborn as sns
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from statannot import add_stat_annotation
from sklearn.linear_model import LinearRegression




Path = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Paires_1.5vs4mMCa'
file_morpho_terminals_20Hz = pd.read_excel(r"\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Tidy_files\GluSnFR_avg_clustering_variables_all_profiles_1.5_4mM_Ca_paired_filtered_3sigma.xlsx", sheet_name='Bouton_size_20Hz')
file_morpho_terminals_50Hz = pd.read_excel(r"\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Tidy_files\GluSnFR_avg_clustering_variables_all_profiles_1.5_4mM_Ca_paired_filtered_3sigma.xlsx", sheet_name='Bouton_size_50Hz')
filter_value = 9
save_figures = False
folder_to_save_figures = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Paires_1.5vs4mMCa\Figures'
file_fold_increase_ephy = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Controls\Controle_electrophy_Ca\A1_fold_change.xlsx'


def palette_colors(num):
    clr = []
    cmap = cm.Blues(np.linspace(0.2,0.85,num))
    for c in range(num):
        rgba = cmap[c]
        clr.append(colors.rgb2hex(rgba))
    return clr


files = sorted(os.listdir(Path))

boutons_15 = {}
boutons_4 = {}

amplitudes_15 = {}
amplitudes_4 = {}

boutons_15_no_fail = {}
boutons_4_no_fail = {}

amp1_15_no_fail = {}
amp1_4_no_fail = {}

AMP1_NoFAIL_15_20Hz = []
ALL_AMPS_NoFAIL_4_20Hz = []

TAU_15, TAU_4 = [],[]
PERC_FAIL_15, PERC_FAIL_4 = [],[]
PERC_FAIL_15_20Hz, PERC_FAIL_15_50Hz = [],[]
AMPS_15, AMPS_4 = [],[]

for file in tqdm(files):
    if 'converted' in file:
        if 'converted.xlsx' in file:
            profile = pd.read_excel('{}/{}'.format(Path, file), sheet_name='Traces DF_F0')
            file = file.rsplit('_',1)[0]
            if '1.5mMCa' in file:
                trace_15 = profile.iloc[:,-2:]
                boutons_15[f'{file}'] = trace_15

            elif '4mMCa' in file:
                trace_4 = profile.iloc[:,-2:]
                boutons_4[f'{file}'] = trace_4


        elif 'Amp.xlsx' in file:
            mean = pd.read_excel('{}/{}'.format(Path, file))
            file = file.rsplit('_',1)[0]
            if '1.5mMCa' in file:
                amps_15 = mean.iloc[:,1:4]
                amplitudes_15[f'{file}'] = amps_15
                if '20Hz' in file:
                    amps_15_ = mean.iloc[:,1:]
                    AMPS_15.append(amps_15_)
                
            elif '4mMCa' in file:
                amps_4 = mean.iloc[:,1:4]
                amplitudes_4[f'{file}'] = amps_4
                if '20Hz' in file:
                    amps_4_ = mean.iloc[:,1:]
                    AMPS_4.append(amps_4_)
        
        elif 'Tau' in file:
            data_tau = pd.read_excel('{}/{}'.format(Path, file))
            if '1.5mMCa' in file:
                tau_a1 = data_tau.iloc[0,1]*1000
                TAU_15.append(tau_a1)
            if '4mMCa' in file:
                tau_a1 = data_tau.iloc[0,1]*1000
                TAU_4.append(tau_a1)
                
                
        elif 'data_bootstrap.xlsx' in file:
            data_failure = pd.read_excel('{}/{}'.format(Path, file), sheet_name='PEAK1')
            if '1.5mMCa' in file:
                perc_fail_a1 = data_failure.iloc[0,-1]
                PERC_FAIL_15.append(perc_fail_a1)
                if '20Hz' in file:
                    PERC_FAIL_15_20Hz.append(perc_fail_a1)
                elif '50Hz' in file:
                    PERC_FAIL_15_50Hz.append(perc_fail_a1)
            elif '4mMCa' in file:
                perc_fail_a1 = data_failure.iloc[0,-1]
                PERC_FAIL_4.append(perc_fail_a1)
        
        
        elif 'NoFail.xlsx' in file:
            data_no_fail = pd.read_excel('{}/{}'.format(Path, file), sheet_name='Traces DF_F0')
            file = file.rsplit('_',1)[0]
            if '1.5mMCa' in file:
                trace_15_no_fail = data_no_fail.iloc[:,-2:]
                boutons_15_no_fail[f'{file}'] = trace_15_no_fail
            
            elif '4mMCa' in file:
                trace_4_no_fail = data_no_fail.iloc[:,-2:]
                boutons_4_no_fail[f'{file}'] = trace_4_no_fail
            
            
        elif 'NoFail_Amp1.xlsx' in file:
            data_amps_no_fail = pd.read_excel('{}/{}'.format(Path, file))
            if '1.5mMCa' in file:
                if '20Hz' in file:
                    AMP1_NoFAIL_15_20Hz.append(data_amps_no_fail.iloc[0,1])
                amp1_no_fail = data_amps_no_fail.iloc[0,1]
                amp1_15_no_fail[f'{file}'] = amp1_no_fail
                
                
            elif '4mMCa' in file:
                if '20Hz' in file:
                    ALL_AMPS_NoFAIL_4_20Hz.append(data_amps_no_fail.iloc[0,1:])
                amp1_no_fail = data_amps_no_fail.iloc[0,1]
                amp1_4_no_fail[f'{file}'] = amp1_no_fail
                
                
                


#########################
#### TRACES FIGURES #####
#########################
            
######## TRACES + AMPLITUDES WITH FAILURES ###########

TRACES_A1_15, TRACES_A1_4 = [],[]         
AMP1_15, AMP1_4 = [], []
RATIO_4_15 = []
PPR_15_20Hz, PPR_4_20Hz = [], []
PPR_15_50Hz, PPR_4_50Hz = [], []

fig_amp1, ax_amp1 = plt.subplots(1,2,sharey=True)
x = np.linspace(0.49, 0.52, num=27, endpoint=True)
for (name_trace_15, trace_15), (name_trace_4, trace_4), (name_amp_15, amp_15), (name_amp_4, amp_4) in zip(boutons_15.items(), boutons_4.items(), amplitudes_15.items(), amplitudes_4.items()):
    name = name_trace_15.rsplit('_', 4)
    name = name[0] + '_' + name[3]
    
    amp1_15 = amp_15.iloc[0,0]
    amp1_4 = amp_4.iloc[0,0]
    ppr_15 = amp_15.iloc[0,1]/amp1_15
    ppr_4 = amp_4.iloc[0,1]/amp1_4
    
    AMP1_15.append(amp1_15)
    AMP1_4.append(amp1_4)
    if '20Hz' in name:
        PPR_15_20Hz.append(ppr_15)
        PPR_4_20Hz.append(ppr_4)
    
    elif '50Hz' in name:
        PPR_15_50Hz.append(ppr_15)
        PPR_4_50Hz.append(ppr_4)

    
    fig_traces, ax_traces = plt.subplots(1,2, figsize=(14,5), tight_layout=True)
    ax_traces[0].plot(trace_4['Time'], savgol_filter(trace_4['Average'], filter_value, 2), 'r', label='Ca 4mM')
    ax_traces[0].plot(trace_15['Time'], savgol_filter(trace_15['Average'], filter_value, 2), 'k', label='Ca 1.5mM')
    ax_traces[0].set_xlabel('#Stim')
    ax_traces[0].set_ylabel('DF/F')
    ax_traces[0].set_title(f'{name}')
    ax_traces[0].legend()
    
    ratio = amp_4/amp_15
    ratio = ratio.transpose()
    RATIO_4_15.append(ratio.iloc[0,0])
    
    ratio.plot(kind='bar', ax=ax_traces[1], stacked=False, rot=2)
    ax_traces[1].axhline(y=1.0, color='k', linestyle='--')
    ax_traces[1].set_ylabel('Ratio')
    ax_traces[1].set_title('Ratio 4mM/1.5mM')
    
    
    start_15 = np.ravel(np.where(trace_15['Time'] >= 0.49))[0]
    stop_15 = np.ravel(np.where(trace_15['Time'] <= 0.52))[-1]
    
    start_4 = np.ravel(np.where(trace_4['Time'] >= 0.49))[0]
    stop_4 = np.ravel(np.where(trace_4['Time'] <= 0.52))[-1]
    
    a1_15 = savgol_filter(trace_15['Average'][start_15:stop_15], filter_value, 2)
    a1_4 = savgol_filter(trace_4['Average'][start_4:stop_4], filter_value, 2)
    resampling15 = resample(a1_15, 27)
    resampling4 = resample(a1_4, 27)
    
    TRACES_A1_15.append(resampling15)
    TRACES_A1_4.append(resampling4)
    
    ax_amp1[0].plot(x, resampling15, 'k', lw=3, alpha=0.1)
    ax_amp1[1].plot(x, resampling4, 'r', lw=3, alpha=0.1)
    ax_amp1[0].set_ylabel('DF/F0')
    ax_amp1[0].set_xlabel('Time (sec)')
    ax_amp1[0].set_title('AMP1 1.5mM Ca')
    ax_amp1[1].set_title('AMP1 4mM Ca')
    
    
    if save_figures == True:
        fig_traces.savefig(f'{folder_to_save_figures}\{name}.pdf')


ax_amp1[0].plot(x, np.mean(TRACES_A1_15, axis=0), 'k', lw=3)
ax_amp1[1].plot(x, np.mean(TRACES_A1_4, axis=0), 'r', lw=3)
       
        

######## TRACES + AMPLITUDES WITHOUT FAILURES ###########

TRACES_A1_NoFAIL_15, TRACES_A1_NoFAIL_4 = [],[]
AMP1_15_No_FAIL, AMP1_4_No_FAIL = [],[]
RATIO_4_15_NoFAIL = []

fig_amp1_no_fail, ax_amp1_no_fail = plt.subplots(1,2,sharey=True)
for (name_trace_15_no_fail, trace_15_no_fail), (name_trace_4_no_fail, trace_4_no_fail), (name_amp1_no_fail_15, amp1_no_fail_15), (name_amp1_no_fail_4, amp1_no_fail_4) in zip(boutons_15_no_fail.items(), boutons_4_no_fail.items(), amp1_15_no_fail.items(), amp1_4_no_fail.items()):
    name_no_fail = name_trace_15_no_fail.rsplit('_', 5)
    name_no_fail = name_no_fail[0] + '_' + name_no_fail[3] + '_NoFailAmp1'
    
    AMP1_15_No_FAIL.append(amp1_no_fail_15)
    AMP1_4_No_FAIL.append(amp1_no_fail_4)
    
    fig_no_fail, ax_no_fail = plt.subplots(1,2, figsize=(14,5), tight_layout=True)
    ax_no_fail[0].plot(trace_4_no_fail['Time'], savgol_filter(trace_4_no_fail['Average'], filter_value, 2), 'r', label='Ca 4mM')
    ax_no_fail[0].plot(trace_15_no_fail['Time'], savgol_filter(trace_15_no_fail['Average'], filter_value, 2), 'k', label='Ca 1.5mM')
    ax_no_fail[0].set_xlabel('#Stim')
    ax_no_fail[0].set_ylabel('DF/F')
    ax_no_fail[0].set_title(f'{name_no_fail}')
    ax_no_fail[0].legend()
    
    ratio_amp1 = amp1_no_fail_4/amp1_no_fail_15
    RATIO_4_15_NoFAIL.append(ratio_amp1)
    
    pd.Series(ratio_amp1).plot(kind='bar', ax=ax_no_fail[1], stacked=False, rot=2)
    ax_no_fail[1].axhline(y=1.0, color='k', linestyle='--')
    ax_no_fail[1].set_ylabel('Ratio')
    ax_no_fail[1].set_title('Ratio 4mM/1.5mM AMP1')
    
    start_15_no_fail = np.ravel(np.where(trace_15_no_fail['Time'] >= 0.49))[0]
    stop_15_no_fail = np.ravel(np.where(trace_15_no_fail['Time'] <= 0.52))[-1]
    
    start_4_no_fail = np.ravel(np.where(trace_4_no_fail['Time'] >= 0.49))[0]
    stop_4_no_fail = np.ravel(np.where(trace_4_no_fail['Time'] <= 0.52))[-1]
    
    a1_NoFail_15 = savgol_filter(trace_15_no_fail['Average'][start_15_no_fail:stop_15_no_fail], filter_value, 2)
    a1_NoFail_4 = savgol_filter(trace_4_no_fail['Average'][start_4_no_fail:stop_4_no_fail], filter_value, 2)
    resampling15 = resample(a1_NoFail_15, 27)
    resampling4 = resample(a1_NoFail_4, 27)
    
    TRACES_A1_NoFAIL_15.append(resampling15)
    TRACES_A1_NoFAIL_4.append(resampling4)
    
    ax_amp1_no_fail[0].plot(x, resampling15, 'k', lw=3, alpha=0.1)
    ax_amp1_no_fail[1].plot(x, resampling4, 'r', lw=3, alpha=0.1)
    ax_amp1_no_fail[0].set_ylabel('DF/F0')
    ax_amp1_no_fail[0].set_xlabel('Time (sec)')
    ax_amp1_no_fail[0].set_title('AMP1 No FAIL 1.5mM Ca')
    ax_amp1_no_fail[1].set_title('AMP1 No Fail 4mM Ca')
    
    if save_figures == True:
        fig_no_fail.savefig(f'{folder_to_save_figures}\{name_no_fail}_NoFail.pdf')
    
ax_amp1_no_fail[0].plot(x, np.mean(TRACES_A1_NoFAIL_15, axis=0), 'k', lw=3)
ax_amp1_no_fail[1].plot(x, np.mean(TRACES_A1_NoFAIL_4, axis=0), 'r', lw=3)


################## DATAFRAMES ####################

df_amps = pd.concat((pd.DataFrame(AMP1_15), pd.DataFrame(AMP1_4), pd.DataFrame(AMP1_15_No_FAIL), pd.DataFrame(AMP1_4_No_FAIL)), axis=1)
df_amps.columns = ['1.5mM', '4mM', '1.5mM_NoFail', '4mM_NoFail']

df_tau = pd.concat((pd.DataFrame(TAU_15), pd.DataFrame(TAU_4)), axis=1)
df_tau.columns = ['Tau_1.5mM', 'Tau_4mM']

df_fold = pd.concat((pd.DataFrame(RATIO_4_15), pd.DataFrame(RATIO_4_15_NoFAIL)), axis=1)
df_fold.columns = ['Ratio', 'Ratio_NoFail']

df_perc_failures = pd.concat((pd.DataFrame(PERC_FAIL_15), pd.DataFrame(PERC_FAIL_4)), axis=1)
df_perc_failures.columns = ['%Fail_1.5mM', '%Fail_4mM']

df_ppr = pd.concat((pd.DataFrame(PPR_15_20Hz), pd.DataFrame(PPR_4_20Hz), pd.DataFrame(PPR_15_50Hz), pd.DataFrame(PPR_4_50Hz)), axis=1)
df_ppr.columns = ['PPR_1.5mM_20Hz', 'PPR_4mM_20Hz', 'PPR_1.5mM_50Hz', 'PPR_4mM_50Hz']

df_fail_freq = pd.concat((pd.DataFrame(PERC_FAIL_15_20Hz), pd.DataFrame(PERC_FAIL_15_50Hz)), axis=1)
df_fail_freq.columns = ['%Fail_20Hz','%Fail_50Hz']

df_size_terminals = pd.concat((file_morpho_terminals_20Hz['Area (µm2)'], file_morpho_terminals_50Hz['Area (µm2)']), axis=1)
df_size_terminals.columns = ['Area_20Hz', 'Area_50Hz']

clr = palette_colors(df_ppr.shape[1])
stat_test = 'Wilcoxon'


#########################
###### AMP1 VIOLIN ######
#########################
        
fig_hist, ax_hist = plt.subplots()
sns.violinplot(data=df_amps.drop(['1.5mM_NoFail', '4mM_NoFail'], axis=1), binwidth=0.1, palette=[clr[0], clr[1]], ax=ax_hist, stat='density')
ax_hist.set_xlabel('DF/F0')
ax_hist.set_title('AMP1 distribution')


#########################
# AMP1  NO-FAIL VIOLIN ##
#########################

fig_hist_NoFail, ax_hist_NoFail = plt.subplots()
sns.histplot(data=df_amps.drop(['1.5mM', '4mM'], axis=1).dropna(), binwidth=0.1, palette=['grey', 'r'], ax=ax_hist_NoFail, stat='density')
ax_hist_NoFail.set_xlabel('DF/F0')
ax_hist_NoFail.set_title('AMP1 No Fail distribution')

fig_violin_amp1, ax_violin_amp1 = plt.subplots(1,2, figsize=(16,6))
sns.violinplot(data=df_amps, ax=ax_violin_amp1[0], palette=['grey', 'r', 'grey', 'r'])
[ax_violin_amp1[0].plot([0,1], [df_amps['1.5mM'][i], df_amps['4mM'][i]], color='grey', marker='o', alpha=0.2) for i in range(df_amps.shape[0])]
[ax_violin_amp1[0].plot([2,3], [df_amps['1.5mM_NoFail'][i], df_amps['4mM_NoFail'][i]], color='grey', marker='o', alpha=0.2) for i in range(df_amps.shape[0])]
add_stat_annotation(ax_violin_amp1[0], data=df_amps, box_pairs=[('1.5mM', '4mM'),
                                                             ('1.5mM_NoFail', '4mM_NoFail')], test=stat_test, text_format='full', verbose=2)
ax_violin_amp1[0].set_ylabel('DF/F0')
ax_violin_amp1[0].set_title('AMP1')

amps_eff_size = dabest.load(df_amps, idx=(('1.5mM', '4mM'),
                                          ('1.5mM_NoFail', '4mM_NoFail')), resamples=5000)
amps_eff_size.mean_diff.plot(ax=ax_violin_amp1[1], custom_palette=['grey', 'r', 'grey', 'r'])


#########################
###### TAU A1 PLOT ######
#########################

fig_tau, ax_tau = plt.subplots()
sns.violinplot(data=df_tau, ax=ax_tau, palette=['grey', 'r'])
sns.swarmplot(data=df_tau, ax=ax_tau, palette=['grey', 'grey'])
add_stat_annotation(ax_tau, data=df_tau, box_pairs=[('Tau_1.5mM', 'Tau_4mM')], test='t-test_ind', text_format='full', verbose=2)
ax_tau.set_ylabel('Decay-time (ms)')
ax_tau.set_title('Tau A1 for 1.5mM and 4mM')


#########################
#### RATIO AMP1 PLOT ####
#########################

fig_fold, ax_fold = plt.subplots()
sns.violinplot(data=df_fold, ax=ax_fold, palette='ch:.25')
sns.swarmplot(data=df_fold, ax=ax_fold, palette=['grey', 'grey'])
add_stat_annotation(ax_fold, data=df_fold, box_pairs=[('Ratio', 'Ratio_NoFail')], test='Mann-Whitney', text_format='full', verbose=2)
ax_fold.set_ylabel('Ratio')
ax_fold.set_title('Ratio of A1_4mM / A1_1.5mM')


#########################
### FAILURES A1 PLOT ####
#########################

fig_perc_failures, ax_perc_failures = plt.subplots()
Psyn = 1 - (df_perc_failures/100)
sns.violinplot(data=Psyn, ax=ax_perc_failures, palette=[clr[0], clr[1]])
[ax_perc_failures.plot([0,1], [Psyn['%Fail_1.5mM'][i], Psyn['%Fail_4mM'][i]], color='grey', marker='o', alpha=0.2) for i in range(Psyn.shape[0])]
add_stat_annotation(ax_perc_failures, data=Psyn, box_pairs=[('%Fail_1.5mM', '%Fail_4mM')], test=stat_test, text_format='full', verbose=2)
ax_perc_failures.set_ylabel('A1 failures (%)')
ax_perc_failures.set_title('Failures for A1 at 1.5mM and 4mM')


################################
##### QUANTAL PRAMAS PLOT ######
################################

fig_N_sites, ax_N_sites = plt.subplots(1,3,tight_layout=True)
x_cumulative = np.arange(1,11)
df_ratio = pd.DataFrame(index=None, columns=None)
p_15 = [1 - (PERC_FAIL_15[i]/100) for i in range(len(PERC_FAIL_15))]
p_4 = [1 - (PERC_FAIL_4[i]/100) for i in range(len(PERC_FAIL_4))]
q = df_amps['4mM_NoFail'].dropna()/df_amps['1.5mM_NoFail'].dropna()
N = []
for i in range(len(AMP1_NoFAIL_15_20Hz)):
    
    ratio = [ALL_AMPS_NoFAIL_4_20Hz[i][k]/AMP1_NoFAIL_15_20Hz[i] for k in range(len(ALL_AMPS_NoFAIL_4_20Hz[i]))]
    df_ratio[i] = ratio
    cumulative = np.cumsum(ratio)
    N.append(cumulative/cumulative[0])
    # ax_N_sites[0].plot(x_cumulative, ratio)
    ax_N_sites[2].plot(x_cumulative, cumulative/cumulative[0], 'r', alpha=0.1)

mean_cum = np.mean(N, axis=0)
sem_cum = stats.sem(N, axis=0)

sns.histplot(data=p_15, ax=ax_N_sites[0], color='k', binwidth=0.07, stat='probability', alpha=0.8)
sns.histplot(data=p_4, ax=ax_N_sites[0], color='r', binwidth=0.07, stat='probability', alpha=1)
sns.histplot(data=q, ax=ax_N_sites[1], color='r', binwidth=0.8, stat='probability', alpha=1)
# ax_N_sites[0].set_ylabel('An 4mM / A1 1.5mM')
# ax_N_sites[0].set_ylim(0, np.max(np.max(df_ratio))+1)
ax_N_sites[0].set_xlabel('Psyn')
ax_N_sites[1].set_xlabel('q')
ax_N_sites[1].set_xlim(0, 14)
ax_N_sites[2].plot(x_cumulative, mean_cum, 'r', marker='o')
ax_N_sites[2].fill_between(x_cumulative, mean_cum+sem_cum, mean_cum-sem_cum, color='r', alpha=0.3)
ax_N_sites[2].set_ylabel('Cumulative quantal release')
ax_N_sites[2].set_xlabel('Number stim')
ax_N_sites[2].set_ylim(0,20)

# [ax_N_sites[0].plot(x_cumulative, df_ratio.transpose().iloc[i,:], color='r', marker='o', alpha=0.2) for i in range(df_ratio.transpose().shape[0])]
# ax_N_sites[0].plot(x_cumulative, np.mean(df_ratio.transpose(), axis=0), color='r', marker='o', lw=2)
# ax_N_sites[0].axhline(y=1., color='k', ls='--')




#########################
### PPR2/1 VIOLINPLOT ###
#########################

fig_violin, ax_violin = plt.subplots(1,2, figsize=(16,6))
sns.violinplot(data=df_ppr, ax=ax_violin[0], palette=['grey', 'r', 'grey', 'r'])
[ax_violin[0].plot([0,1], [df_ppr['PPR_1.5mM_20Hz'][i], df_ppr['PPR_4mM_20Hz'][i]], color='grey', marker='o', alpha=0.2) for i in range(df_ppr.shape[0])]
[ax_violin[0].plot([2,3], [df_ppr['PPR_1.5mM_50Hz'][i], df_ppr['PPR_4mM_50Hz'][i]], color='grey', marker='o', alpha=0.2) for i in range(df_ppr.shape[0])]
add_stat_annotation(ax_violin[0], data=df_ppr, box_pairs=[('PPR_1.5mM_20Hz', 'PPR_4mM_20Hz'),
                                                          ('PPR_1.5mM_50Hz', 'PPR_4mM_50Hz')], test=stat_test, text_format='full', verbose=2)
ax_violin[0].set_ylabel('PPR')
ax_violin[0].set_title('PPR2/1')

ppr_eff_size = dabest.load(df_ppr, idx=(('PPR_1.5mM_20Hz', 'PPR_4mM_20Hz'),
                                        ('PPR_1.5mM_50Hz', 'PPR_4mM_50Hz')), resamples=5000)
ppr_eff_size.mean_diff.plot(ax=ax_violin[1], custom_palette=['grey', 'r', 'grey', 'r'])


slope_20Hz = [df_ppr['PPR_4mM_20Hz'][i] - df_ppr['PPR_1.5mM_20Hz'][i] for i in range(df_ppr.shape[0])]
slope_50Hz = [df_ppr['PPR_4mM_50Hz'][i] - df_ppr['PPR_1.5mM_50Hz'][i] for i in range(df_ppr.shape[0])]

#PPR_slope vs PPR
fig_slope_ppr, ax_slope_ppr = plt.subplots()
ax_slope_ppr.scatter(df_ppr['PPR_1.5mM_20Hz'], slope_20Hz, color='k', alpha=0.5)
# ax_slope_ppr.scatter(df_ppr['PPR_1.5mM_50Hz'], slope_50Hz, color='k')

X = np.array(df_ppr['PPR_1.5mM_20Hz']).reshape(-1,1)
reg = LinearRegression().fit(X, slope_20Hz)
pred = reg.predict(X)
r2 = reg.score(X,slope_20Hz)
ax_slope_ppr.plot(X, pred, color='k')

ax_slope_ppr.set_xlabel('PPR 1.5mM')
ax_slope_ppr.set_ylabel('Slope (PPR4-PPR1.5)')
ax_slope_ppr.set_title(f'R2 = {r2}')

Psyn_15_20Hz = 1 - (df_fail_freq['%Fail_20Hz']/100)
Psyn_15_50Hz = 1 - (df_fail_freq['%Fail_50Hz']/100)

#Slope vs PPR 1.5mM vs Psyn
plt.figure()
ax_plot3d = plt.axes(projection='3d')
[ax_plot3d.scatter3D(df_ppr['PPR_1.5mM_20Hz'][i], slope_20Hz[i], Psyn_15_20Hz[i], color='k', alpha=0.4) for i in range(df_ppr.shape[0])]
ax_plot3d.set_xlabel('PPR')
ax_plot3d.set_ylabel('Slope')
ax_plot3d.set_zlabel('Psyn')

#Boutons area vs PPR 1.5mM
plt.figure()
# ax_plot3d_morpho = plt.axes(projection='3d')
# [ax_plot3d_morpho.scatter3D(df_ppr['PPR_1.5mM_20Hz'][i], df_size_terminals['Area_20Hz'][i], Psyn_15_20Hz[i], color='k', alpha=0.4) for i in range(df_ppr.shape[0])]
[plt.scatter(df_ppr['PPR_1.5mM_20Hz'].dropna()[i], df_size_terminals['Area_20Hz'][i], color='k') for i in range(df_ppr.shape[0])]
plt.ylim(0,2.5)
plt.xlabel('PPR 1.5mM 20Hz')
plt.ylabel('Terminal area (µm2)')

##################################
### FOLD INCREASE EPHY/GLUSNFR ###
##################################

fold_ephy = pd.read_excel(file_fold_increase_ephy)
fold_change_glusnfr_with_failures = pd.DataFrame([AMP1_4[i]/AMP1_15[i] for i in range(len(AMP1_15))])
df_fold_change = pd.concat([fold_ephy, fold_change_glusnfr_with_failures], axis=1)
df_fold_change.columns = ['Ephy', 'iGluSnFR.S72A']
fig_fold_change, ax_fold_change = plt.subplots()
sns.violinplot(data=df_fold_change, ax=ax_fold_change, palette=['grey','g'])                       
add_stat_annotation(ax_fold_change, data=df_fold_change, box_pairs=[('Ephy', 'iGluSnFR.S72A')], test='Mann-Whitney', text_format='full', verbose=2)

######## SAVE FIGURES ###
#########################

if save_figures == True:
    fig_amp1.savefig(f'{folder_to_save_figures}\AMP1_overlaped_traces_1.5vs4mMCa.pdf')
    fig_amp1_no_fail.savefig(f'{folder_to_save_figures}\AMP1_overlaped_traces_1.5vs4mMCa_NoFail.pdf')
    fig_tau.savefig(f'{folder_to_save_figures}\Tau_A1.pdf')
    fig_hist.savefig(f'{folder_to_save_figures}\AMP1_histogram.pdf')
    fig_hist_NoFail.savefig(f'{folder_to_save_figures}\AMP1_NoFail_histogram.pdf')
    fig_violin_amp1.savefig(f'{folder_to_save_figures}\AMP1_violin.pdf')
    fig_fold.savefig(f'{folder_to_save_figures}\Ratio_A1_4mM_1.5mM.pdf')
    fig_perc_failures.savefig(f'{folder_to_save_figures}\PercFail_A1.pdf')
    fig_N_sites.savefig(f'{folder_to_save_figures}\Quantal_params.pdf')
    fig_violin.savefig(f'{folder_to_save_figures}\PPR_20Hz_50Hz.pdf')
