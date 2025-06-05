# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 16:04:03 2021

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


Path = r'E:\AAVDJ.GluSnFR-S72A\Controls\iGluSnFRvsEphy\Saturation and desensitization\20211221_linescan1_20Hz_3pulses_4mM'
linescan = '20211221'


files = sorted(os.listdir(Path))



REC_EPHY = []
REC_KYN_CYCLO_EPHY = []
TRACE_GLUSNFR = []
TRACE_KYN_CYCLO_GLUSNFR = []

Amplitudes = {}
Tau = {}


for file in range(len(files)):
    if linescan in files[file]:
        if '.wcp' in files[file]:
            ephy_file = io.WinWcpIO(f'{Path}/{files[file]}')
            bl = ephy_file.read_block()
            
            for episode in bl.segments :
                time_ephy = episode.analogsignals[0].times #The time vector
                
                rec = episode.analogsignals[0].magnitude #The signal vector
                x1 = np.ravel(np.where(time_ephy >= 0.1))[0]
                x2 = np.ravel(np.where(time_ephy <= 0.15))[-1]
                leak = np.mean(rec[x1:x2], axis=0)
                rec = rec-leak
                
                if '4mMCa.wcp' in files[file]:
                    REC_EPHY.append(rec)
                else:
                    REC_KYN_CYCLO_EPHY.append(rec)
        
        
        elif '4mMCa_converted.xlsx' in files[file]:
            trace = pd.read_excel(f'{Path}/{files[file]}', sheet_name = 'Traces DF_F0')
            trace = trace.drop(['Average'], axis=1)
            time = trace['Time']
            
            f0 = pd.read_excel(f'{Path}/{files[file]}', sheet_name = 'F0')
            for i in range(len(trace.columns)):
                TRACE_GLUSNFR.append(trace.iloc[:,i])
        
        elif '4mMCa_kyn_cyclo_converted.xlsx' in files[file]:
            trace_kyn_cyclo = pd.read_excel(f'{Path}/{files[file]}', sheet_name = 'Traces DF_F0')
            trace_kyn_cyclo = trace_kyn_cyclo.drop(['Average'], axis=1)
            time_kyn_cyclo = trace_kyn_cyclo['Time']
            
            f0_kyn_ctz = pd.read_excel(f'{Path}/{files[file]}', sheet_name = 'F0')
            for i in range(len(trace_kyn_cyclo.columns)):
                TRACE_KYN_CYCLO_GLUSNFR.append(trace_kyn_cyclo.iloc[:,i])
            
        
        elif '4mMCa_Ephy_Amp' in files[file]:
            amp_ephy = pd.read_excel(f'{Path}/{files[file]}')
            Amplitudes['AMP_EPHY'] = amp_ephy
        
        elif 'kyn_cyclo_Ephy_Amp' in files[file]:
            amp_ephy_kyn_cyclo = pd.read_excel(f'{Path}/{files[file]}')
            Amplitudes['AMP_EPHY_KYN_CTZ'] = amp_ephy_kyn_cyclo
        
        elif '4mMCa_converted_Amp' in files[file]:
            amp_glusnfr = pd.read_excel(f'{Path}/{files[file]}')
            Amplitudes['AMP_GLUSNFR'] = amp_glusnfr
            
        elif 'kyn_cyclo_converted_Amp' in files[file]:
            amp_glusnfr_kyn_cyclo = pd.read_excel(f'{Path}/{files[file]}')
            Amplitudes['AMP_GLUSNFR_KYN_CTZ'] = amp_glusnfr_kyn_cyclo
        
        elif '4mMCa_Ephy_Tau' in files[file]:
            tau = pd.read_excel(f'{Path}/{files[file]}')
            Tau['TAU_EPHY'] = tau
            
        elif '4mMCa_converted_Tau' in files[file]:
            tau = pd.read_excel(f'{Path}/{files[file]}')
            Tau['TAU_GLUSNFR'] = tau
        
        elif 'kyn_cyclo_Ephy_Tau' in files[file]:
            tau = pd.read_excel(f'{Path}/{files[file]}')
            Tau['TAU_EPHY_KYN_CTZ'] = tau
        
        elif 'kyn_cyclo_converted_Tau' in files[file]:
            tau = pd.read_excel(f'{Path}/{files[file]}')
            Tau['TAU_GLUSNFR_KYN_CTZ'] = tau
            

###### F0 BEFORE VS AFTER KYN+CTZ ############### 

plt.figure()
plt.scatter(np.arange(1,f0.iloc[0,:-1].shape[0]+1), f0.iloc[0,:-1].values, color='g')
plt.scatter(np.arange(f0.iloc[0,:-1].shape[0]+1, f0.iloc[0,:-1].shape[0]+1 + f0_kyn_ctz.iloc[0,:-1].shape[0]), f0_kyn_ctz.iloc[0,:-1].values, color='purple')
plt.scatter(int(len(f0.iloc[0,:-1])/2), f0['Average'], marker='*', color='g', s=200)
plt.scatter(len(f0.iloc[0,:-1])+int(len(f0_kyn_ctz.iloc[0,:-1])/2), f0_kyn_ctz['Average'], marker='*', color='purple', s=200)
plt.ylabel('F0 (A.U.)')
plt.xlabel('#Ep')
plt.ylim(0)

###### EPISODES WITHOUT KYN AND CTZ ############### 

for i in range(len(REC_EPHY)):
    fig, ax = plt.subplots(1,4,figsize=(15,4),tight_layout=True)
    ax[0].plot(time_ephy, REC_EPHY[i], 'gray')
    ax[0].set_ylabel('Amplitude (pA)')
    
    ax2 = ax[0].twinx()
    ax3 = ax[1].twinx()
    mpl_axes_aligner.align.yaxes(ax[0], 0, ax2, 0, 0.5)
    # ax2.plot(time, savgol_filter(TRACE_GLUSNFR[i], 9, 2), 'g')
    ax2.plot(time, TRACE_GLUSNFR[i], 'g')
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_ylabel('DF/F')
    ax2.yaxis.label.set_color('g')
    ax2.tick_params(axis='y', colors='g')
    ax2.spines['right'].set_color('g')
    
    ax3.yaxis.label.set_color('g')
    ax3.tick_params(axis='y', colors='g')
    ax3.spines['right'].set_color('g')
    
    for (name_amps, dataset_amps), (name_tau, dataset_tau) in zip(Amplitudes.items(), Tau.items()):
        if 'KYN_CTZ' not in name_amps:
            amps = dataset_amps.iloc[i,1:]
            ppr21 = amps[1]/amps[0]
            ppr32 = amps[2]/amps[1]
            tau = dataset_tau.iloc[i,1:]*1000
            
            if 'EPHY' in name_amps:
                ax[1].bar([0,2,4], [abs(amps[j]) for j in range(3)], color='grey')
                ax[2].bar([0,2], [ppr21, ppr32], color='grey')
                ax[3].bar([0,2,4], [tau[j] for j in range(3)], color='grey')
            elif 'GLUSNFR' in name_amps:
                ax3.bar([1,3,5], [abs(amps[j]) for j in range(3)], color='g') 
                ax[2].bar([1,3], [ppr21, ppr32], color='g')
                ax[3].bar([1,3,5], [tau[j] for j in range(3)], color='g')
            
            
###### EPISODES WITH KYN AND CTZ ###############       

for i in range(len(REC_KYN_CYCLO_EPHY)):
    fig, ax = plt.subplots(1,4,figsize=(15,4),tight_layout=True)
    ax[0].plot(time_ephy, REC_KYN_CYCLO_EPHY[i], 'k')
    ax[0].set_ylabel('Amplitude (pA)')
    
    ax2 = ax[0].twinx()
    ax3 = ax[1].twinx()
    mpl_axes_aligner.align.yaxes(ax[0], 0, ax2, 0, 0.5)
    # ax2.plot(time, savgol_filter(TRACE_KYN_CYCLO_GLUSNFR[i], 9, 2), 'purple')
    ax2.plot(time, TRACE_KYN_CYCLO_GLUSNFR[i], 'purple')
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_ylabel('DF/F')
    ax2.yaxis.label.set_color('purple')
    ax2.tick_params(axis='y', colors='purple')
    ax2.spines['right'].set_color('purple')
    
    ax3.yaxis.label.set_color('g')
    ax3.tick_params(axis='y', colors='g')
    ax3.spines['right'].set_color('g')
    
    for (name_amps, dataset_amps), (name_tau, dataset_tau) in zip(Amplitudes.items(), Tau.items()):
        if 'KYN_CTZ' in name_amps:
            amps = dataset_amps.iloc[i,1:]
            ppr21 = amps[1]/amps[0]
            ppr32 = amps[2]/amps[1]
            tau = dataset_tau.iloc[i,1:]*1000
        
            if 'EPHY' in name_amps:
                ax[1].bar([0,2,4], [abs(amps[j]) for j in range(3)], color='k')
                ax[2].bar([0,2], [ppr21, ppr32], color='k')
                ax[3].bar([0,2,4], [tau[j] for j in range(3)], color='k')
            elif 'GLUSNFR' in name_amps:
                ax3.bar([1,3,5], [abs(amps[j]) for j in range(3)], color='purple') 
                ax[2].bar([1,3], [ppr21, ppr32], color='purple')
                ax[3].bar([1,3,5], [tau[j] for j in range(3)], color='purple')



###### GLOBAL FIGURE ###############

fig_glob, ax_glob = plt.subplots(1,4,figsize=(15,4),tight_layout=True)
ax_glob[0].plot(time_ephy, np.mean(REC_EPHY, axis=0), color='grey', label='EPHY')
ax_glob[0].plot(time_ephy, np.mean(REC_KYN_CYCLO_EPHY, axis=0), color='k', label='EPHY + KYN/CTZ')
ax_glob[0].set_ylabel('Amplitude (pA)')
    
ax2_glob = ax_glob[0].twinx()
ax3_glob = ax_glob[1].twinx()
mpl_axes_aligner.align.yaxes(ax_glob[0], 0, ax2_glob, 0, 0.5)

ax2_glob.plot(time, np.mean(TRACE_GLUSNFR, axis=0), color='g', label='GLUSNFR')
ax2_glob.plot(time, np.mean(TRACE_KYN_CYCLO_GLUSNFR, axis=0), color='purple', label='GLUSNFR + KYN/CTZ')
ax2_glob.set_ylim(-1.5, 1.5)
ax2_glob.set_ylabel('DF/F')
ax2_glob.yaxis.label.set_color('g')
ax2_glob.tick_params(axis='y', colors='g')
ax2_glob.spines['right'].set_color('g')

ax3.yaxis.label.set_color('g')
ax3.tick_params(axis='y', colors='g')
ax3.spines['right'].set_color('g')



amps_ephy = Amplitudes['AMP_EPHY'].iloc[:,1:]
amps_ephy_kyn_ctz = Amplitudes['AMP_EPHY_KYN_CTZ'].iloc[:,1:]
amps_glusnfr = Amplitudes['AMP_GLUSNFR'].iloc[:,1:]
amps_glusnfr_kyn_ctz = Amplitudes['AMP_GLUSNFR_KYN_CTZ'].iloc[:,1:]

ppr21_ephy = amps_ephy['AMP2']/amps_ephy['AMP1']
ppr32_ephy = amps_ephy['AMP3']/amps_ephy['AMP2']

ppr21_ephy_kyn_ctz = amps_ephy_kyn_ctz['AMP2']/amps_ephy_kyn_ctz['AMP1']
ppr32_ephy_kyn_ctz = amps_ephy_kyn_ctz['AMP3']/amps_ephy_kyn_ctz['AMP2']

ppr21_glusnfr = amps_glusnfr['AMP2']/amps_glusnfr['AMP1']
ppr32_glusnfr = amps_glusnfr['AMP3']/amps_glusnfr['AMP2']

ppr21_glusnfr_kyn_ctz = amps_glusnfr_kyn_ctz['AMP2']/amps_glusnfr_kyn_ctz['AMP1']
ppr32_glusnfr_kyn_ctz = amps_glusnfr_kyn_ctz['AMP3']/amps_glusnfr_kyn_ctz['AMP2']

tau_ephy = Tau['TAU_EPHY'].iloc[:,1:]*1000
tau_glusnfr = Tau['TAU_GLUSNFR'].iloc[:,1:]*1000
tau_ephy_kyn_ctz = Tau['TAU_EPHY_KYN_CTZ'].iloc[:,1:]*1000
tau_glusnfr_kyn_ctz = Tau['TAU_GLUSNFR_KYN_CTZ'].iloc[:,1:]*1000


ax_glob[1].bar([0,2,4], [abs(np.mean(amps_ephy['AMP1'])), abs(np.mean(amps_ephy['AMP2'])), abs(np.mean(amps_ephy['AMP3']))], color='grey', alpha=0.5)
[ax_glob[1].scatter([0,2,4], [abs(amps_ephy['AMP1'][i]), abs(amps_ephy['AMP2'][i]), abs(amps_ephy['AMP3'][i])], color='grey') for i in range(amps_ephy.shape[0])]
ax3_glob.bar([1,3,5], [np.mean(amps_glusnfr['AMP1']), np.mean(amps_glusnfr['AMP2']), np.mean(amps_glusnfr['AMP3'])], color='g', alpha=0.5)
[ax3_glob.scatter([1,3,5], [amps_glusnfr['AMP1'][i], amps_glusnfr['AMP2'][i], amps_glusnfr['AMP3'][i]], color='g') for i in range(amps_glusnfr.shape[0])]


ax_glob[1].bar([7,9,11], [abs(np.mean(amps_ephy_kyn_ctz['AMP1'])), abs(np.mean(amps_ephy_kyn_ctz['AMP2'])), abs(np.mean(amps_ephy_kyn_ctz['AMP3']))], color='k', alpha=0.5)
[ax_glob[1].scatter([7,9,11], [abs(amps_ephy_kyn_ctz['AMP1'][i]), abs(amps_ephy_kyn_ctz['AMP2'][i]), abs(amps_ephy_kyn_ctz['AMP3'][i])], color='k') for i in range(amps_ephy_kyn_ctz.shape[0])]
ax3_glob.bar([8,10,12], [np.mean(amps_glusnfr_kyn_ctz['AMP1']), np.mean(amps_glusnfr_kyn_ctz['AMP2']), np.mean(amps_glusnfr_kyn_ctz['AMP3'])], color='purple', alpha=0.5)
[ax3_glob.scatter([8,10,12], [amps_glusnfr_kyn_ctz['AMP1'][i], amps_glusnfr_kyn_ctz['AMP2'][i], amps_glusnfr_kyn_ctz['AMP3'][i]], color='purple') for i in range(amps_glusnfr_kyn_ctz.shape[0])]


ax_glob[2].bar([0,1,3,4], [np.mean(ppr21_ephy), np.mean(ppr21_glusnfr), np.mean(ppr21_ephy_kyn_ctz), np.mean(ppr21_glusnfr_kyn_ctz)], color=['grey', 'g', 'k', 'purple'], alpha=0.5)
[ax_glob[2].scatter([0,1,3,4], [ppr21_ephy[i], ppr21_glusnfr[i], ppr21_ephy_kyn_ctz[i], ppr21_glusnfr_kyn_ctz[i]], color=['grey', 'g', 'k', 'purple']) for i in range(ppr21_ephy.shape[0])]
[ax_glob[2].plot([0,1], [ppr21_ephy[i], ppr21_glusnfr[i]], color='k') for i in range(ppr21_ephy.shape[0])]
[ax_glob[2].plot([3,4], [ppr21_ephy_kyn_ctz[i], ppr21_glusnfr_kyn_ctz[i]], color='k') for i in range(ppr21_ephy_kyn_ctz.shape[0])]



ax_glob[3].bar([0,2,4], [abs(np.mean(tau_ephy['AMP1'])), abs(np.mean(tau_ephy['AMP2'])), abs(np.mean(tau_ephy['AMP3']))], color='grey', alpha=0.5)
[ax_glob[3].scatter([0,2,4], [abs(tau_ephy['AMP1'][i]), abs(tau_ephy['AMP2'][i]), abs(tau_ephy['AMP3'][i])], color='grey') for i in range(tau_ephy.shape[0])]
ax_glob[3].bar([1,3,5], [np.mean(tau_glusnfr['AMP1']), np.mean(tau_glusnfr['AMP2']), np.mean(tau_glusnfr['AMP3'])], color='g', alpha=0.5)
[ax_glob[3].scatter([1,3,5], [tau_glusnfr['AMP1'][i], tau_glusnfr['AMP2'][i], tau_glusnfr['AMP3'][i]], color='g') for i in range(tau_glusnfr.shape[0])]


ax_glob[3].bar([7,9,11], [abs(np.mean(tau_ephy_kyn_ctz['AMP1'])), abs(np.mean(tau_ephy_kyn_ctz['AMP2'])), abs(np.mean(tau_ephy_kyn_ctz['AMP3']))], color='k', alpha=0.5)
[ax_glob[3].scatter([7,9,11], [abs(tau_ephy_kyn_ctz['AMP1'][i]), abs(tau_ephy_kyn_ctz['AMP2'][i]), abs(tau_ephy_kyn_ctz['AMP3'][i])], color='k') for i in range(tau_ephy_kyn_ctz.shape[0])]
ax_glob[3].bar([8,10,12], [np.mean(tau_glusnfr_kyn_ctz['AMP1']), np.mean(tau_glusnfr_kyn_ctz['AMP2']), np.mean(tau_glusnfr_kyn_ctz['AMP3'])], color='purple', alpha=0.5)
[ax_glob[3].scatter([8,10,12], [tau_glusnfr_kyn_ctz['AMP1'][i], tau_glusnfr_kyn_ctz['AMP2'][i], tau_glusnfr_kyn_ctz['AMP3'][i]], color='purple') for i in range(tau_glusnfr_kyn_ctz.shape[0])]


    