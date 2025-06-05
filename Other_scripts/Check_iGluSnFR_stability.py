# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 17:57:47 2022

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import seaborn as sns
import dabest
import numpy as np
import pandas as pd
import os
from math import ceil
from scipy.signal import savgol_filter
from statannot import add_stat_annotation
from sklearn.linear_model import LinearRegression
import itertools

Path = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Boutons_analysis'
Save_folder = r'E:\AAVDJ.GluSnFR-S72A\Paires_1.5vs4mMCa\All_traces_and_amps_paired_1.5vs4_20Hz_and_50Hz\Figures_individual_traces'
freq = '20Hz'
calcium = '2.5mMCa'
stat_test = 'Mann-Whitney'
save = False


if freq == '20Hz':
    peak_start = 0.548
    peak_stop = 0.56

elif freq == '50Hz':
    peak_start = 0.518
    peak_stop = 0.53


files = sorted(os.listdir(Path))
F0_SINGLE, F0_TRIAL_AVERAGED = [],[]
AMP2_SINGLE = []
for file in range(len(files)):
    if freq in files[file]:
        if calcium in files[file]:
            if 'traces.xlsx' in files[file]:
                print(files[file])
                traces = pd.read_excel(f'{Path}\{files[file]}', sheet_name='Traces')
                fback = pd.read_excel(f'{Path}\{files[file]}', sheet_name='Data').dropna()
                fback = fback.iloc[-1,1:]
                time = traces['Time']
                traces = traces.drop('Time', axis=1) - fback
                traces = [savgol_filter(traces[i], 9, 2) for i in range(traces.shape[1])]
                
                f0_start = np.ravel(np.where(time >= 0.38))[0]
                f0_stop = np.ravel(np.where(time <= 0.48))[-1]
                f0_mean = [np.mean(traces[i][f0_start:f0_stop]) for i in range(len(traces))]
                F0_TRIAL_AVERAGED.append(f0_mean)
                f0_std = [np.std(traces[i][f0_start:f0_stop]) for i in range(len(traces))]

                df_f0 = [(traces[i] - f0_mean[i])/f0_mean[i] for i in range(len(traces))]
                x1 = np.ravel(np.where(time >= peak_start))[0]
                x2 = np.ravel(np.where(time <= peak_stop))[-1]
                amp2_df_f = [np.max(df_f0[i][x1:x2]) for i in range(len(df_f0))]
                idx = [time[x1:x2].values[np.argmax(df_f0[i][x1:x2])] for i in range(len(df_f0))]
                
                for i in range(len(f0_mean)):
                    F0_SINGLE.append(f0_mean[i])
                    AMP2_SINGLE.append(amp2_df_f[i])
                
                amp2 = [np.max(traces[i][x1:x2]) for i in range(len(traces))]
                z_scores = [(amp2[i]-f0_mean[i])/f0_std[i] for i in range(len(amp2))]
                
                i = 0
                j = 0
                # fig_traces, ax_traces = plt.subplots(3, ceil(len(traces)/3), figsize=(14,6), sharey=True)
                # fig_df_f0, ax_df_f0 = plt.subplots(3, ceil(len(traces)/3), figsize=(14,6), sharey=True)
                
                # for item in range(len(traces)):
                #     if j == ceil(len(traces)/3):
                #         i += 1
                #         j = 0
                            
                #     elif j == ceil(2*(len(traces)/3)):
                #         i += 1
                #         j = 0
                    
                #     ax_traces[i,j].plot(time, traces[item], 'k')
                #     ax_traces[i,j].scatter(idx[item], amp2[item], color='r')
                #     ax_df_f0[i,j].plot(time, df_f0[item], 'r')
                #     ax_df_f0[i,j].scatter(idx[item], amp2_df_f[item], color='k')
                    
                #     j += 1
                    
                # ax_traces[0,0].set_title(f'{files[file]}')
                # ax_df_f0[0,0].set_title(f'{files[file]}')
                
                
                fig_amp2_ep, ax_amp2_ep = plt.subplots(1,3,figsize=(12,4),tight_layout=True)
                var = [amp2_df_f, f0_mean, f0_mean]
                regr = LinearRegression()
                for i in range(len(var)):
                    if i == 1:
                        X = np.array(var[i]).reshape(-1,1)
                        Y = var[i-1]
                    else:
                        X = np.array(range(len(var[i]))).reshape(-1,1)
                        Y = var[i]
                        
                    regr.fit(X, Y)
                    r2 = regr.score(X, Y)
                    r = np.sqrt(r2)
                    regr_pred = regr.predict(X)
                    
                    ax_amp2_ep[i].scatter(X, Y, color='k', alpha=0.5)
                    ax_amp2_ep[i].plot(X, regr_pred, 'k')
                    
                    ax_amp2_ep[i].text(np.max(X),0,'R = {:.3f}\nR2 = {:.3f}'.format(r,r2))
                    ax_amp2_ep[i].set_ylim(0, np.max(Y)+(np.max(Y)/3))
                    
                
                ax_amp2_ep[0].set_ylabel('Amp2 (DF/F)')
                ax_amp2_ep[0].set_xlabel('#Trial')
                
                ax_amp2_ep[1].set_title(f'{files[file]}')
                ax_amp2_ep[1].set_ylabel('Amp2 (DF/F)')
                ax_amp2_ep[1].set_xlabel('F0 (A.U.)')
                
                ax_amp2_ep[2].set_ylabel('F0 (A.U.)')
                ax_amp2_ep[2].set_xlabel('#Trial')
                
                # # name = files[file].rsplit('_',1)
                # if save == True:
                #     fig_traces.savefig(f'{Save_folder}/{name[0]}_raw_traces.pdf')
                #     fig_df_f0.savefig(f'{Save_folder}/{name[0]}_df_f_traces.pdf')
                #     fig_amp2_ep.savefig(f'{Save_folder}/{name[0]}_Amp2_regression.pdf')
                
                
                
                # df_gr1 = traces.iloc[:,0:int(traces.shape[1]/2)]
                # df_gr2 = traces.iloc[:,int(traces.shape[1]/2):int(traces.shape[1]+1)]
                
                # mean_gr1 = savgol_filter(np.mean(df_gr1, axis=1), 9, 2)
                # mean_gr2 = savgol_filter(np.mean(df_gr2, axis=1), 9, 2)
                # df_mean = pd.concat((pd.DataFrame(mean_gr1), pd.DataFrame(mean_gr2)), axis=1)
                # df_mean.columns = ['MEAN GR1', 'MEAN_GR2']
                
                # f0_mean_gr1 = np.mean(mean_gr1[f0_start:f0_stop], axis=0)
                # f0_mean_gr2 = np.mean(mean_gr2[f0_start:f0_stop], axis=0)
                
                # A = np.vstack([mean_gr1, np.ones(len(mean_gr1))]).T
                # m,c = np.linalg.lstsq(A, mean_gr2, rcond=None)[0]
                
                # fig, ax = plt.subplots(1,4,figsize=(10,4),tight_layout=True)
                # ax[0].plot(mean_gr1, mean_gr2)
                # ax[0].plot(mean_gr1, m * mean_gr1 + c, 'r', label='{:.2f}x+{:.2f}'.format(m,c))
                # ax[0].legend()

                # [ax[1].plot(time, traces[i], 'k', alpha=0.3) for i in range(traces.shape[1])]
                # ax[1].plot(time, mean_gr1, 'r', label='Gr1')
                # ax[1].plot(time, mean_gr2, 'b', label='Gr2')
                # ax[1].legend()
                
                # ax[2].plot(time, mean_gr1-f0_mean_gr1, 'r', label='Gr1')
                # ax[2].plot(time, mean_gr2-f0_mean_gr2, 'b', label='Gr2')
                # ax[2].legend()
                
                # effect_size = dabest.load(df_mean, idx=('MEAN GR1', 'MEAN_GR2'), resamples=5000)
                # effect_size.mean_diff.plot(ax=ax[3], custom_palette=['r', 'b'])
                # effect_size.mean_diff.statistical_tests





x = np.array(F0_SINGLE).reshape(-1,1)
regr = LinearRegression()
regr.fit(x, AMP2_SINGLE)
r2 = regr.score(x, AMP2_SINGLE)
r = np.sqrt(r2)
regr_pred = regr.predict(x)

df = pd.concat((pd.DataFrame(F0_SINGLE, columns=['F0']), pd.DataFrame(AMP2_SINGLE, columns=['Amp2 (DF/F)'])), axis=1)

fig, ax = plt.subplots(1,2,figsize=(12,4),tight_layout=True)
sns.histplot(data=df['F0'], binwidth=2, color='darkgreen', ax=ax[0], stat='probability')
sns.regplot(x='F0', y='Amp2 (DF/F)', data=df, ci=95, ax=ax[1], color='darkgreen', scatter_kws = {'alpha': 0.1}, line_kws = {'color':'g'})
ax[1].errorbar(df['F0'].mean(), df['Amp2 (DF/F)'].mean(), xerr=df['F0'].std(), yerr=df['Amp2 (DF/F)'].std(), fmt='o', color='r')
ax[1].axhline(y=0, ls='--', color='k')
ax[1].text(np.max(F0_SINGLE),np.min(AMP2_SINGLE), 'R = {:.3f}\nR2 = {:.3f}'.format(r,r2))