# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:31:09 2021

@author: Theo.ROSSI
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neo import io
import os
from scipy.signal import savgol_filter
import mpl_axes_aligner
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statannot import add_stat_annotation


Path = r'E:\AAVDJ.GluSnFR-S72A\Controls\iGluSnFRvsEphy'
linescan = '20211013_linescan1'

files = sorted(os.listdir(Path))


#########################
#### FILES EXTRACTION ###
#########################

REC_EPHY, TIME_EPHY = [],[]

for file in files:
    if linescan in file:
        if 'Traces' in file:
            glusnfr_file = pd.read_excel(f'{Path}/{file}')
            glusnfr_avg = glusnfr_file['Average']
            time_glusnfr = glusnfr_file['Time']
            x_start = np.ravel(np.where(time_glusnfr >= 0.45))[0]
            traces = glusnfr_file.drop(['Average', 'Time'], axis=1)
        
        
        elif 'dF_F0_Amp' in file:
            glusnfr_amp = pd.read_excel(f'{Path}/{file}')
            glusnfr_avg_amp = glusnfr_amp.iloc[-1,:]
            glusnfr_amps = glusnfr_amp.iloc[:-1,1:]
            glusnfr_ppr = [glusnfr_amps.iloc[:,i]/glusnfr_amps.iloc[:,0] for i in range(len(glusnfr_amps.columns))]
        
        
        elif 'ephy_Amp' in file:
            ephy_amp = pd.read_excel(f'{Path}/{file}')
            ephy_amps = ephy_amp.iloc[:, 1:]
            ephy_ppr = [ephy_amps.iloc[:,i]/ephy_amps.iloc[:,0] for i in range(len(ephy_amps.columns))]
            
            
        elif 'ephy' in file:
            ephy_file = io.WinWcpIO(f'{Path}/{file}')
            bl = ephy_file.read_block()
            
            for episode in bl.segments :
                time_ephy = episode.analogsignals[0].times #The time vector
                TIME_EPHY.append(time_ephy)
                
                rec = episode.analogsignals[0].magnitude #The signal vector
                x1 = np.ravel(np.where(time_ephy >= 0.3))[0]
                x2 = np.ravel(np.where(time_ephy <= 0.4))[-1]
                x3 = np.ravel(np.where(time_ephy <= 1.1))[-1]
                leak = np.mean(rec[x1:x2], axis=0)
                rec = rec-leak
                REC_EPHY.append(rec)
            
            ephy_avg = np.mean(REC_EPHY, axis=0)
    
    
    

#########################
##### TRACES PLOTS ######
#########################

for i in range(len(REC_EPHY)):
    fig, ax = plt.subplots(figsize=(4,3), tight_layout=True)
    ax.plot(TIME_EPHY[i][x2:x3], REC_EPHY[i][x2:x3], 'k')
    ax.set_ylabel('Amplitude (pA)')
    ax.set_xlabel('Time (sec)')
    ax2 = ax.twinx()
    mpl_axes_aligner.align.yaxes(ax, 0, ax2, 0, 0.5)
    ax2.plot(time_glusnfr[x_start:], savgol_filter(traces.iloc[x_start:,i], 7, 2), 'g')
    ax2.set_ylim(-2,2)
    ax2.set_ylabel('DF/F')
    ax2.yaxis.label.set_color('g')
    ax2.tick_params(axis='y', colors='g')
    ax2.spines['right'].set_color('g')



#########################
##### AVG TRACE PLOT ####
#########################
            
fig_avg, ax_avg = plt.subplots(figsize=(4,3), tight_layout=True)

ax_avg.plot(time_ephy[x2:x3], ephy_avg[x2:x3], 'k')
ax_avg.set_ylabel('Amplitude (pA)')
ax_avg.set_xlabel('Time (sec)')
ax_avg2 = ax_avg.twinx()
mpl_axes_aligner.align.yaxes(ax_avg, 0, ax_avg2, 0, 0.5)
ax_avg2.plot(time_glusnfr[x_start:], glusnfr_avg[x_start:], 'g')
ax_avg2.set_ylim(-2,2)
ax_avg2.set_ylabel('DF/F')
ax_avg2.yaxis.label.set_color('g')
ax_avg2.tick_params(axis='y', colors='g')
ax_avg2.spines['right'].set_color('g')



#########################
###### PPR PLOT #########
#########################
    
df_pprA2_A1 = pd.concat([ephy_ppr[1], glusnfr_ppr[1]], axis=1)
df_pprA2_A1.columns = ['Ephy', 'iGluSnFR']

fig_ppr, ax_ppr = plt.subplots(figsize=(3,4), tight_layout=True)
sns.boxplot(data=df_pprA2_A1, showmeans=True, ax=ax_ppr, palette=['#253494', '#7fcdbb'],
            meanprops={'markerfacecolor':'red', 'markeredgecolor':'black', 'markersize':'8'})

[plt.scatter(0, df_pprA2_A1['Ephy'][i], color='k') for i in range(df_pprA2_A1.shape[0])]
[plt.scatter(1, df_pprA2_A1['iGluSnFR'][i], color='k') for i in range(df_pprA2_A1.shape[0])]
[plt.plot([0,1], [df_pprA2_A1['Ephy'][i], df_pprA2_A1['iGluSnFR'][i]], 'k', alpha=0.5) for i in range(df_pprA2_A1.shape[0])]

ax_ppr.set_xticklabels(['Ephy', 'iGluSnFR'])
ax_ppr.set_ylabel('PPR A2/A1')

add_stat_annotation(ax_ppr, data=df_pprA2_A1, box_pairs=[('Ephy', 'iGluSnFR')],
                    test='t-test_welch', text_format='star', verbose=2)



#########################
###### AMP1 PLOT ########
#########################

scaler = MinMaxScaler()
df_amps = pd.concat([ephy_amps.iloc[:,:2], glusnfr_amps.iloc[:,:2]], axis=1)

scaled_df_amps = pd.DataFrame(scaler.fit_transform(df_amps), columns = ['AMP1_ephy', 'AMP2_ephy', 'AMP1_glusnfr', 'AMP2_glusnfr'])

fig_amp, ax_amp = plt.subplots(figsize=(3,4), tight_layout=True)
sns.boxplot(data=scaled_df_amps, showmeans=True, ax=ax_amp, palette=['#253494', '#225ea8', '#7fcdbb', '#c7e9b4'],
            meanprops={'markerfacecolor':'red', 'markeredgecolor':'black', 'markersize':'8'})

# [plt.scatter(i, scaled_df_amps.iloc[j,i], color='k') for i,j in zip(range(len(scaled_df_amps.columns)), range(scaled_df_amps.shape[0]))]

# [plt.plot([0,1], [df_amps['AMP1_ephy'][i], df_amps['AMP2_ephy'][i]], 'k', alpha=0.3) for i in range(df_amps.shape[0])]
# [plt.plot([2,3], [df_amps['AMP1_glusnfr'][i], df_amps['AMP2_glusnfr'][i]], 'k', alpha=0.3) for i in range(df_amps.shape[0])]

ax_amp.set_xticklabels(['A1_Ephy','A2_Ephy','A1_iGluSnFR','A2_iGluSnFR'])
ax_amp.set_ylabel('Norm. amplitude')

# add_stat_annotation(ax_amp, data=scaled_df_amps, box_pairs=[('AMP1_ephy','AMP1_glusnfr'), ('AMP2_ephy','AMP2_glusnfr')],
#                     test='t-test_paired', text_format='star', verbose=2)
