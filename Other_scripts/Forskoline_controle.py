# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:12:25 2022

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter as sf
import scipy.stats as stats
import seaborn as sns
from statannot import add_stat_annotation
from sklearn.mixture import GaussianMixture
import itertools
from tqdm import tqdm

Path = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Controls\iGluSnFR.S72A_forskoline'
folder_to_save = Path+'/'+'Figures'
calcium = '2.5mM'
linescan = '20220421_linescan1'
n_modes = [2,4]
palette = ['k','orange']
save = True

files = sorted(os.listdir(Path))

def gauss_function(x, amp, x0, sigma):
    return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))

NAME = []
TIME_BEFORE, TRACE_BEFORE = [],[]
TIME_AFTER, TRACE_AFTER = [],[]
AMPS_BEFORE, AMPS_AFTER = [],[]
CUMSUM_AMPS_BEFORE, CUMSUM_AMPS_AFTER = [],[]
AMP1_BEFORE, AMP1_AFTER = [],[]
AMP2_BEFORE, AMP2_AFTER = [],[]
NORM_BEFORE, NORM_AFTER = [],[]
PPR_BEFORE, PPR_AFTER = [],[]
FAIL1_BEFORE, FAIL1_AFTER = [],[]
FAIL2_BEFORE, FAIL2_AFTER = [],[]


for file in tqdm(range(len(files))):
    if calcium in files[file]:
        if 'traces_converted.xlsx' in files[file]:
            data = pd.read_excel(f'{Path}\{files[file]}', sheet_name='Traces DF_F0')
            trace = data['Average']
            time = data['Time']
            if 'before' in files[file]:
                name = files[file].rsplit('_',4)[0]
                NAME.append(name)
                TIME_BEFORE.append(time)
                TRACE_BEFORE.append(trace)
            elif 'after' in files[file]:
                TIME_AFTER.append(time)
                TRACE_AFTER.append(trace)
        
        if 'Amp.xlsx' in files[file]:
            data_amp = pd.read_excel(f'{Path}\{files[file]}')
            data_amp = data_amp.iloc[0,1:]
            ppr = [data_amp.iloc[i]/data_amp.iloc[0] for i in range(data_amp.shape[0])]
            if 'before' in files[file]:
                AMPS_BEFORE.append([data_amp[i] for i in range(len(data_amp))])
                AMP1_BEFORE.append(data_amp[0])
                AMP2_BEFORE.append(data_amp[1])
                PPR_BEFORE.append(ppr[1])
                NORM_BEFORE.append(ppr)
            elif 'after' in files[file]:
                AMPS_AFTER.append([data_amp[i] for i in range(len(data_amp))])
                AMP1_AFTER.append(data_amp[0])
                AMP2_AFTER.append(data_amp[1])
                PPR_AFTER.append(ppr[1])
                NORM_AFTER.append(ppr)
        
        if 'data_bootstrap.xlsx' in files[file]:
            data_fail = [pd.read_excel(f'{Path}\{files[file]}', sheet_name=f'PEAK{i}') for i in np.arange(1,4)]
            fail1 = 1-(data_fail[0]['PercFail'][0]/100)
            fail2 = 1-(data_fail[1]['PercFail'][0]/100)
            if 'before' in files[file]:
                FAIL1_BEFORE.append(fail1)
                FAIL2_BEFORE.append(fail2)
            elif 'after' in files[file]:
                FAIL1_AFTER.append(fail1)
                FAIL2_AFTER.append(fail2)
                

AMPS_PF_BEFORE, AMPS_PF_AFTER = [],[]
NORM_PF_BEFORE, NORM_PF_AFTER = [],[]
RATIO = []
RATIO_PF = []             
fig_fold, ax_fold = plt.subplots(1,2)
fig_PF, ax_PF = plt.subplots(1,4,figsize=(14,4),tight_layout=True)
stim_number = np.arange(1,11)
for item in range(len(NAME)):
    ratio = [AMPS_AFTER[item][i]/AMPS_BEFORE[item][i] for i in range(len(AMPS_BEFORE[item]))]
    RATIO.append(ratio)
    # ratio_after = [AMPS_AFTER[item][i]/AMPS_BEFORE[item][0] for i in range(len(AMPS_AFTER[item]))]
    # cumulative_before = np.cumsum(ratio_before)
    # cumulative_after = np.cumsum(ratio_after)
    
    ax_fold[0].plot(np.arange(1,11), ratio, palette[1], marker='o', alpha=0.2)
    ax_fold[1].scatter([0,1], [AMPS_BEFORE[item][0]/AMPS_BEFORE[item][0], ratio[1]], color=[palette[0],palette[1]], lw=2, marker='o')
    ax_fold[1].plot([0,1], [AMPS_BEFORE[item][0]/AMPS_BEFORE[item][0], ratio[1]], color='k', lw=2)
    # ax_fold.plot(np.arange(1,11), cumulative_after, palette[1], marker='o', alpha=0.5)
    
    if linescan in NAME[item]:
        RATIO_PF.append(ratio)
        AMPS_PF_BEFORE.append(AMPS_BEFORE[item])
        AMPS_PF_AFTER.append(AMPS_AFTER[item])
        NORM_PF_BEFORE.append(NORM_BEFORE[item])
        NORM_PF_AFTER.append(NORM_AFTER[item])
        ax_PF[0].plot(stim_number, NORM_BEFORE[item], palette[0], lw=2, alpha=0.2)
        ax_PF[0].plot(stim_number, NORM_AFTER[item], palette[1], lw=2, alpha=0.2)
        ax_PF[1].plot(stim_number, ratio, palette[1], lw=2, marker='o', alpha=0.2)
        ax_PF[3].scatter([0,1], [AMPS_BEFORE[item][0]/AMPS_BEFORE[item][0], ratio[1]], color=[palette[0],palette[1]], lw=2, marker='o')
        ax_PF[3].plot([0,1], [AMPS_BEFORE[item][0]/AMPS_BEFORE[item][0], ratio[1]], color='k', lw=2)
        # ax_PF[1].plot(stim_number, cumulative_after, palette[1], lw=2, marker='o', alpha=0.5)
        
    fig, ax = plt.subplots(1,3,figsize=(18,5))
    ax[0].plot(TIME_BEFORE[item], sf(TRACE_BEFORE[item],9,2), palette[0], label=f'Ca {calcium}')
    ax[0].plot(TIME_AFTER[item], sf(TRACE_AFTER[item],9,2), palette[1], label=f'Ca {calcium} + Forsk 50µM')
    ax[0].set_title(NAME[item])
    ax[0].axhline(y=0.0, color='k', ls='--')
    ax[0].legend()
    ax[1].plot(stim_number, NORM_BEFORE[item], palette[0], marker='o')
    ax[1].plot(stim_number, NORM_AFTER[item], palette[1], marker='o')
    ax[1].axhline(y=1.0, color='k', ls='--')
    # ax[2].plot(stim_number, cumulative_before, palette[0], marker='o')
    ax[2].plot(stim_number, ratio, palette[1], marker='o')
    
    if save == True:
        plt.savefig(f'{folder_to_save}/{NAME[item]}.pdf')

mean_norm_before = np.mean(NORM_PF_BEFORE, axis=0)
sem_norm_before = stats.sem(NORM_PF_BEFORE, axis=0)

mean_norm_after = np.mean(NORM_PF_AFTER, axis=0)
sem_norm_after = stats.sem(NORM_PF_AFTER, axis=0)

ax_fold[0].plot(np.arange(1,11), np.mean(RATIO, axis=0), color='orange', marker='o', lw=3)
ax_fold[0].axhline(y=1., color='k', ls='--')
ax_fold[0].set_ylabel('Cumulative ratio (An/A1)')
ax_fold[0].set_xlabel('Stim number')
ax_fold[0].legend([f'Ca {calcium}', f'Ca {calcium} + FSK (50µM)'])

ax_PF[0].plot(stim_number, mean_norm_before, palette[0], marker='o')
ax_PF[0].plot(stim_number, mean_norm_after, palette[1], marker='o')
ax_PF[0].fill_between(stim_number, mean_norm_before+sem_norm_before, mean_norm_before-sem_norm_before, palette[0])
ax_PF[0].fill_between(stim_number, mean_norm_after+sem_norm_after, mean_norm_after-sem_norm_after, palette[1])
ax_PF[0].set_title(linescan)
ax_PF[0].axhline(y=1., color='k', ls='--')
ax_PF[0].set_ylabel('An/A1')
ax_PF[0].set_xlabel('Stim number')
ax_PF[0].legend([f'Ca {calcium}', f'Ca {calcium} + FSK (50µM)'])

ax_PF[1].plot(stim_number, np.mean(RATIO_PF, axis=0), palette[1], marker='o')
ax_PF[1].axhline(y=1., color='k', ls='--')
ax_PF[1].set_ylabel('Ratio (An FSK / An {calcium})')
ax_PF[1].set_xlabel('Stim number')
ax_PF[1].set_ylim(0,5)

ax_PF[3].set_ylabel('Norm. A1')
ax_PF[3].set_ylim(0,4)




df_pf_amps = pd.concat((pd.DataFrame(list(itertools.chain.from_iterable(AMPS_PF_BEFORE))),
                        pd.DataFrame(list(itertools.chain.from_iterable(AMPS_PF_AFTER)))), axis=1)
df_pf_amps.columns = ['Before_FSK','After_FSK']

sns.histplot(data=df_pf_amps, ax=ax_PF[2], binwidth=0.03, palette=palette, stat='density', alpha=0.5)

for item in range(df_pf_amps.shape[1]):  
    gmm = GaussianMixture(n_components=n_modes[item], covariance_type="full", tol=0.001)
    gmm_x = np.linspace(0, np.max(np.max(df_pf_amps)), 5000)
    
    gmm = gmm.fit(X=np.expand_dims(df_pf_amps.iloc[:,item].dropna(), 1))
    
    print(f'{df_pf_amps.columns[item]} Modes:')
    print(gmm.means_)
    print('----------------')
    
    
    gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

    # Construct function manually as sum of gaussians
    gmm_y_sum = np.full_like(gmm_x, fill_value=0, dtype=np.float32)
    for m, c, w in zip(gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()):
        gmm_y_sum += gauss_function(x=gmm_x, amp=w, x0=m, sigma=np.sqrt(c))

    # Normalize so that integral is 1    
    gmm_y_sum /= np.trapz(gmm_y_sum, gmm_x)

    ax_PF[2].plot(gmm_x, gmm_y, color=palette[item], lw=2, label=df_pf_amps.columns[item])





df = pd.concat((pd.DataFrame(AMP1_BEFORE), pd.DataFrame(AMP1_AFTER),
               pd.DataFrame(AMP2_BEFORE), pd.DataFrame(AMP2_AFTER),
               pd.DataFrame(PPR_BEFORE), pd.DataFrame(PPR_AFTER),
               pd.DataFrame(FAIL1_BEFORE), pd.DataFrame(FAIL1_AFTER),
               pd.DataFrame(FAIL2_BEFORE), pd.DataFrame(FAIL2_AFTER)), axis=1)
df.columns = ['AMP1', 'AMP1_FSK', 'AMP2', 'AMP2_FSK', 'PPR', 'PPR_FSK', 'FAIL1', 'FAIL1_FSK', 'FAIL2', 'FAIL2_FSK']

fig, ax = plt.subplots(1,3, figsize=(10,4), tight_layout=True)

sns.boxplot(data=df.iloc[:,:4], ax=ax[0], palette=['grey','orange','grey','orange'], showmeans=True)
sns.boxplot(data=df.iloc[:,4:6], ax=ax[1], palette=['grey','orange'], showmeans=True)
sns.boxplot(data=df.iloc[:,6:], ax=ax[2], palette=['grey','orange','grey','orange'], showmeans=True)

ax[0].plot([0,1], [df['AMP1'], df['AMP1_FSK']], 'k', marker='o', alpha=0.2)
ax[0].plot([2,3], [df['AMP2'], df['AMP2_FSK']], 'k', marker='o', alpha=0.2)
ax[1].plot([0,1], [df['PPR'], df['PPR_FSK']], 'k', marker='o', alpha=0.2)
ax[2].plot([0,1], [df['FAIL1'], df['FAIL1_FSK']], 'k', marker='o', alpha=0.2)
ax[2].plot([2,3], [df['FAIL2'], df['FAIL2_FSK']], 'k', marker='o', alpha=0.2)

ax[0].set_ylabel('DF/F')
ax[1].set_ylabel('PPR (A2/A1)')
ax[2].set_ylabel('% failures')
ax[0].set_ylim(0)
ax[1].set_ylim(0)
ax[2].set_ylim(0)

ax[1].set_title(f'Before vs After FSK 50µM at {calcium}')


add_stat_annotation(ax[0], data=df.iloc[:,:4], box_pairs=[('AMP1', 'AMP1_FSK'),
                                                          ('AMP2', 'AMP2_FSK')],
                    test='Wilcoxon', text_format='full', verbose=2)

add_stat_annotation(ax[1], data=df.iloc[:,4:6], box_pairs=[('PPR', 'PPR_FSK')],
                    test='Wilcoxon', text_format='full', verbose=2)

add_stat_annotation(ax[2], data=df.iloc[:,6:], box_pairs=[('FAIL1', 'FAIL1_FSK'),
                                                          ('FAIL2', 'FAIL2_FSK')],
                    test='t-test_paired', text_format='full', verbose=2)







fig_ppr, ax_ppr = plt.subplots()
x = np.arange(1,11)

mean_ppr_before = np.mean([AMPS_BEFORE[i]/AMPS_BEFORE[i][0] for i in range(len(AMPS_BEFORE))], axis=0)
mean_ppr_after = np.mean([AMPS_AFTER[i]/AMPS_AFTER[i][0] for i in range(len(AMPS_AFTER))], axis=0)

sem_ppr_before = stats.sem([AMPS_BEFORE[i]/AMPS_BEFORE[i][0] for i in range(len(AMPS_BEFORE))])
sem_ppr_after = stats.sem([AMPS_AFTER[i]/AMPS_AFTER[i][0] for i in range(len(AMPS_AFTER))])

[ax_ppr.plot(x, AMPS_BEFORE[i]/AMPS_BEFORE[i][0], 'k', alpha=0.2) for i in range(len(AMPS_BEFORE))]
[ax_ppr.plot(x, AMPS_AFTER[i]/AMPS_AFTER[i][0], 'darkorange', alpha=0.2) for i in range(len(AMPS_AFTER))]
ax_ppr.plot(x, mean_ppr_before, 'k', marker='o')
ax_ppr.plot(x, mean_ppr_after, 'darkorange', marker='o')
ax_ppr.fill_between(x, mean_ppr_before+sem_ppr_before, mean_ppr_before-sem_ppr_before, color='k', alpha=0.5)
ax_ppr.fill_between(x, mean_ppr_after+sem_ppr_after, mean_ppr_after-sem_ppr_after, color='darkorange', alpha=0.5)
ax_ppr.set_ylim(0)