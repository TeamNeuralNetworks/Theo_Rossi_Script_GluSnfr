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

Path = r'E:\AAVDJ.GluSnFR-S72A\Paires_1.5vs4mMCa\All_traces_and_amps_paired_1.5vs4_20Hz_and_50Hz'
Save_folder = r'E:\AAVDJ.GluSnFR-S72A\Paires_1.5vs4mMCa\All_traces_and_amps_paired_1.5vs4_20Hz_and_50Hz\Figures_individual_traces'
freq = '50Hz'
calcium = '1.5mMCa'
stat_test = 'Mann-Whitney'
save = True


if freq == '20Hz':
    peak_start = 0.548
    peak_stop = 0.56

elif freq == '50Hz':
    peak_start = 0.518
    peak_stop = 0.53


files = sorted(os.listdir(Path))

for file in range(len(files)):
    if freq in files[file]:
        if calcium in files[file]:
            if 'traces.xlsx' in files[file]:
                traces = pd.read_excel(f'{Path}\{files[file]}', sheet_name='Traces')
                fback = pd.read_excel(f'{Path}\{files[file]}', sheet_name='Data').dropna()
                fback = fback.iloc[2,1:]
                time = traces['Time']
                traces = traces.drop('Time', axis=1) - fback
                
                f0_start = np.ravel(np.where(time >= 0.35))[0]
                f0_stop = np.ravel(np.where(time <= 0.45))[-1]
                f0_mean = [np.mean(traces.iloc[f0_start:f0_stop, i], axis=0) for i in range(traces.shape[1])]

                df_f0 = [savgol_filter((traces[i] - f0_mean[i])/f0_mean[i], 9, 2) for i in range(traces.shape[1])]
                x1 = np.ravel(np.where(time >= peak_start))[0]
                x2 = np.ravel(np.where(time <= peak_stop))[-1]
                amp2 = [np.max(df_f0[i][x1:x2]) for i in range(len(df_f0))]
                idx = [time[x1:x2].values[np.argmax(df_f0[i][x1:x2])] for i in range(len(df_f0))]
                
                
                i = 0
                j = 0
                fig_traces, ax_traces = plt.subplots(3, ceil(traces.shape[1]/3), figsize=(14,6), sharey=True)
                fig_df_f0, ax_df_f0 = plt.subplots(3, ceil(traces.shape[1]/3), figsize=(14,6), sharey=True)
                fig_amp2_ep, ax_amp2_ep = plt.subplots()
                for item in range(traces.shape[1]):
                    if j == ceil(traces.shape[1]/3):
                        i += 1
                        j = 0
                            
                    elif j == ceil(2*(traces.shape[1]/3)):
                        i += 1
                        j = 0
                    
                    ax_traces[i,j].plot(time, savgol_filter(traces.iloc[:,item], 9, 2), 'k')
                    ax_df_f0[i,j].plot(time, df_f0[item], 'r')
                    ax_df_f0[i,j].scatter(idx[item], amp2[item], color='k')
                    
                    j += 1
                    
                ax_traces[0,0].set_title(f'{files[file]}')
                ax_df_f0[0,0].set_title(f'{files[file]}')
                
                X = np.array(range(len(df_f0))).reshape(-1,1)
                regr = LinearRegression()
                regr.fit(X, amp2)
                regr_pred = regr.predict(X)
                a,b = regr.coef_, regr.intercept_
                
                ax_amp2_ep.scatter(X, amp2, color='k')
                ax_amp2_ep.plot(X, regr_pred, 'r')
                ax_amp2_ep.set_title(f'{files[file]}\n{a}x+{b}')
                ax_amp2_ep.set_ylim(0,)
                
                name = files[file].rsplit('_',1)
                if save == True:
                    fig_traces.savefig(f'{Save_folder}/{name[0]}_raw_traces.pdf')
                    fig_df_f0.savefig(f'{Save_folder}/{name[0]}_df_f_traces.pdf')
                    fig_amp2_ep.savefig(f'{Save_folder}/{name[0]}_Amp2_regression.pdf')
                
                
                
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