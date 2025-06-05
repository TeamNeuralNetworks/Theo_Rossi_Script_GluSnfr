# -*- coding: utf-8 -*-
"""
Created on Sat May 14 20:07:56 2022

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import mpl_axes_aligner
import numpy as np
import pandas as pd
import os
from neo import io
import scipy.stats as stats
from collections import Counter

Path = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Controls\iGluSnFRvsEphy\Saturation and desensitization'

files = sorted(os.listdir(Path))

EPHY_FILES, EPHY_KYN_CTZ_FILES = [],[]

ephy, ephy_kyn_ctz = {},{}
glusnfr, glusnfr_kyn_ctz = {},{}
glusnfr_measures, glusnfr_measures_kyn_ctz = {},{}

palette = ['grey', 'k', 'g', 'purple']
labels = ['Ephy', 'Ephy_kyn_ctz', 'Glusnfr', 'Glusnfr_kyn_ctz']


def add_to(file, data, dic):
    name = file.split('.')[0]
    dic[name] = data
    return name, dic
    


for file in range(len(files)):
    if '.wcp' in files[file]:
        ephy_file = io.WinWcpIO(f'{Path}/{files[file]}')
        bl = ephy_file.read_block()
        
        df_rec = pd.DataFrame(index=None, columns=None)
        
        for episode in range(len(bl.segments)) :
            time_ephy = np.array(bl.segments[episode].analogsignals[0].times) #The time vector
            rec = bl.segments[episode].analogsignals[0].magnitude #The signal vector
            x1 = np.ravel(np.where(time_ephy >= 0.1))[0]
            x2 = np.ravel(np.where(time_ephy <= 0.15))[-1]
            leak = np.mean(rec[x1:x2], axis=0)
            rec = rec-leak
            
            df_rec[f'{episode}'] = rec[:,0]
        
        
        if 'kyn_cyclo' not in files[file]:
            EPHY_FILES.append(files[file])
            add_to(files[file], df_rec, ephy)
        
        else:
            EPHY_KYN_CTZ_FILES.append(files[file])
            add_to(files[file], df_rec, ephy_kyn_ctz)

    
    elif '.xlsx' in files[file]:
        if 'Ephy' in files[file]:
            data = pd.read_excel(f'{Path}/{files[file]}').iloc[:,1:]
            
            if 'Amp' or 'Tau' in files[file]:
                if 'kyn_cyclo' not in files[file]:
                    add_to(files[file], data, ephy)
                else:
                    add_to(files[file], data, ephy_kyn_ctz)

        else:
            if 'traces_converted.xlsx' in files[file]:
                data = [pd.read_excel(f'{Path}/{files[file]}', sheet_name=i) for i in np.arange(1,3)]
                if 'kyn_cyclo' not in files[file]:
                    add_to(files[file], data, glusnfr)
                else:
                    add_to(files[file], data, glusnfr_kyn_ctz)
            
            elif 'Amp' or 'Tau' in files[file]:
                if 'traces.xlsx' in files[file]:
                    continue
                else: 
                    data = pd.read_excel(f'{Path}/{files[file]}')
                    if 'kyn_cyclo' not in files[file]:
                        add_to(files[file], data, glusnfr_measures)
                    else:
                        add_to(files[file], data, glusnfr_measures_kyn_ctz)



df_amps = pd.DataFrame(index=None, columns=labels)
df_ppr = pd.DataFrame(index=None, columns=labels)
df_tau = pd.DataFrame(index=None, columns=labels)




for cell in range(len(EPHY_FILES)):
    name = EPHY_FILES[cell].split('_',2)
    name = name[0] + '_' + name[1]
    
    fig, ax = plt.subplots(1, 5, figsize=(16,4), tight_layout=True)
    ax[0].set_title(name)
    for (name_ephy, data_ephy), (name_ephy_kyn_ctz, data_ephy_kyn_ctz) in zip(ephy.items(), ephy_kyn_ctz.items()):
        if name in name_ephy:
            if 'Amp' in name_ephy:
                mean_amps = [abs(np.mean(data_ephy[col])) for col in data_ephy.columns]
                std_amps = [abs(np.std(data_ephy[col])) for col in data_ephy.columns]
                mean_amps_kyn_ctz = [abs(np.mean(data_ephy_kyn_ctz[col])) for col in data_ephy_kyn_ctz.columns]
                std_amps_kyn_ctz = [abs(np.std(data_ephy_kyn_ctz[col])) for col in data_ephy_kyn_ctz.columns]
                df = pd.concat((pd.DataFrame(mean_amps, columns=[labels[0]]),
                                pd.DataFrame(mean_amps_kyn_ctz, columns=[labels[1]])), axis=1)
                df_amps = df_amps.append(df)
                
                
                ppr = data_ephy['AMP2']/data_ephy['AMP1']
                mean_ppr = np.mean(ppr)
                std_ppr = np.std(ppr)
                ppr_kyn_ctz = data_ephy_kyn_ctz['AMP2']/data_ephy_kyn_ctz['AMP1']
                mean_ppr_kyn_ctz = np.mean(ppr_kyn_ctz)
                std_ppr_kyn_ctz = np.std(ppr_kyn_ctz)
                df = pd.concat((pd.DataFrame(mean_ppr, columns=[labels[0]], index=[f'PPR{cell}']),
                                pd.DataFrame(mean_ppr_kyn_ctz, columns=[labels[1]], index=[f'PPR{cell}'])), axis=1)
                df_ppr = df_ppr.append(df)
                
                
                
                ax[1].bar([0,2,4], mean_amps, yerr=std_amps, color=palette[0], alpha=0.5)
                [ax[1].scatter([0,2,4], abs(data_ephy.iloc[i,:]), color=palette[0]) for i in range(data_ephy.shape[0])]
                ax[1].bar([6,8,10], mean_amps_kyn_ctz, yerr=std_amps_kyn_ctz, color=palette[1], alpha=0.5)
                [ax[1].scatter([6,8,10], abs(data_ephy_kyn_ctz.iloc[i,:]), color=palette[1]) for i in range(data_ephy_kyn_ctz.shape[0])]
                
                ax[2].bar(0, mean_ppr, yerr=std_ppr, color=palette[0], alpha=0.5)
                [ax[2].scatter(0, ppr[i], color=palette[0]) for i in range(len(ppr))]
                ax[2].bar(2, mean_ppr_kyn_ctz, yerr=std_ppr_kyn_ctz, color=palette[1], alpha=0.5)
                [ax[2].scatter(2, ppr_kyn_ctz[i], color=palette[1]) for i in range(len(ppr_kyn_ctz))]
                
                
                
            elif 'Tau' in name_ephy:
                mean_tau = [np.mean(data_ephy[col])*1000 for col in data_ephy.columns]
                std_tau = [np.std(data_ephy[col])*1000 for col in data_ephy.columns]
                mean_tau_kyn_ctz = [np.mean(data_ephy_kyn_ctz[col])*1000 for col in data_ephy_kyn_ctz.columns]
                std_tau_kyn_ctz = [np.std(data_ephy_kyn_ctz[col])*1000 for col in data_ephy_kyn_ctz.columns]
                df = pd.concat((pd.DataFrame(mean_tau, columns=[labels[0]]),
                                pd.DataFrame(mean_tau_kyn_ctz, columns=[labels[1]])), axis=1)
                df_tau = df_tau.append(df)
                
                
                ax[3].bar([0,2,4], mean_tau, yerr=std_tau, color=palette[0], alpha=0.5)
                [ax[3].scatter([0,2,4], data_ephy.iloc[i,:]*1000, color=palette[0]) for i in range(data_ephy.shape[0])]
                ax[3].bar([6,8,10], mean_tau_kyn_ctz, yerr=std_tau_kyn_ctz, color=palette[1], alpha=0.5)
                [ax[3].scatter([6,8,10], data_ephy_kyn_ctz.iloc[i,:]*1000, color=palette[1]) for i in range(data_ephy_kyn_ctz.shape[0])]
                ax[3].set_ylabel('Decay-time (ms)')
                
            else:
                mean_ephy = np.mean(data_ephy, axis=1)
                mean_ephy_kyn_ctz = np.mean(data_ephy_kyn_ctz, axis=1)
                
                ax[0].plot(time_ephy, mean_ephy, palette[0])
                ax[0].plot(time_ephy, mean_ephy_kyn_ctz, palette[1])



    groups = [glusnfr, glusnfr_kyn_ctz]
    measures = [glusnfr_measures, glusnfr_measures_kyn_ctz]
    i = 0
    j = 2
    for group in range(len(groups)):
        mean_f0 = pd.DataFrame(np.mean([data_glusnfr[1] for name_glusnfr, data_glusnfr in groups[group].items() if name in name_glusnfr], axis=0))
    
        mean_traces = pd.DataFrame(np.mean([data_glusnfr[0] for name_glusnfr, data_glusnfr in groups[group].items() if name in name_glusnfr], axis=0))
        mean_cell = mean_traces.iloc[:,:-2].mean(axis=1)
        t = mean_traces.iloc[:,-1]
        
        mean_amps = pd.DataFrame(np.mean([data_measure.iloc[:-1,1:].transpose() for name_measure, data_measure in measures[group].items() if name in name_measure if 'Amp' in name_measure], axis=0))
        mean_mean_amps = mean_amps.mean(axis=1)
        std_mean_amps = mean_amps.std(axis=1)
        df_amps = df_amps.append(pd.DataFrame(mean_mean_amps, columns=[labels[j]]))
        
        ppr = mean_amps.iloc[1,:]/mean_amps.iloc[0,:]
        mean_ppr = np.mean(ppr)
        std_ppr = np.std(ppr)
        df_ppr = df_ppr.append(pd.DataFrame(mean_ppr, columns=[labels[j]], index=[f'PPR{cell}']))
        
        mean_tau = pd.DataFrame(np.mean([data_measure.iloc[:-1,1:].transpose() for name_measure, data_measure in measures[group].items() if name in name_measure if 'Tau' in name_measure], axis=0))*1000
        mean_mean_tau = mean_tau.mean(axis=1)
        std_mean_tau = mean_tau.std(axis=1)
        df_tau = df_tau.append(pd.DataFrame(mean_mean_tau, columns=[labels[j]]))
        
        
        ax2 = ax[0].twinx()
        ax3 = ax[1].twinx()
        mpl_axes_aligner.align.yaxes(ax[0], 0, ax2, 0, 0.5)
        
        ax2.plot(t, mean_cell, palette[group+2])
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_ylabel('DF/F')
        ax2.yaxis.label.set_color(palette[2])
        ax2.tick_params(axis='y', colors=palette[2])
        ax2.spines['right'].set_color(palette[2])
        
        ax3.bar([1+i,3+i,5+i], np.array(mean_mean_amps), yerr=std_mean_amps, color=palette[group+2], alpha=0.5)
        [ax3.scatter([1+i,3+i,5+i], mean_amps.iloc[:,k], color=palette[j]) for k in range(mean_amps.shape[1])]
        
        ax[2].bar([j+group-1], mean_ppr, yerr=std_ppr, color=palette[j], alpha=0.5)
        [ax[2].scatter([j+group-1], ppr[k], color=palette[j]) for k in range(ppr.shape[0])]
        
        ax[3].bar([1+i,3+i,5+i], np.array(mean_mean_tau), yerr=std_mean_tau, color=palette[group+2], alpha=0.5)
        [ax[3].scatter([1+i,3+i,5+i], mean_tau.iloc[:,k], color=palette[j]) for k in range(mean_tau.shape[1])]
        
        ax[4].scatter(np.arange(1,mean_f0.size), mean_f0.iloc[:,:-1], color=palette[j])
        
        i += 6
        j += 1






fig_global, ax_global = plt.subplots(1,3, figsize=(10,4), tight_layout=True)
for col in range(df_ppr.shape[1]):
    mean = np.mean(df_ppr.iloc[:,col].dropna())
    std = np.std(df_ppr.iloc[:,col].dropna())
    ax_global[1].bar(col, mean, yerr=std, color=palette[col], alpha=0.5)
    [ax_global[1].scatter(col, df_ppr.iloc[:,col].dropna()[i], color=palette[col]) for i in range(df_ppr.iloc[:,col].dropna().shape[0])]


dataframes = [df_amps, df_tau]
indexes = [0,1,2]
j = 0
for frame in range(len(dataframes)):
    if frame == 0:
        ax1 = ax_global[0].twinx()
        ax1.set_ylabel('DF/F')
        ax1.yaxis.label.set_color(palette[2])
        ax1.tick_params(axis='y', colors=palette[2])
        ax1.spines['right'].set_color(palette[2])
    else:
        ax1 = ax_global[frame+j]
        
    for col in range(len(labels)):
        values = [[dataframes[frame][labels[col]].dropna()[index] for index in dataframes[frame][labels[col]].dropna().index if index == num] for num in indexes]
        print(values)
        means = [np.mean([dataframes[frame][labels[col]].dropna()[index] for index in dataframes[frame][labels[col]].dropna().index if index == num]) for num in indexes]
        std = [np.std([dataframes[frame][labels[col]].dropna()[index] for index in dataframes[frame][labels[col]].dropna().index if index == num]) for num in indexes]
        if col >= 2:
            ax1.bar([6+col,8+col,10+col], means, yerr=std, color=palette[col], alpha=0.5)
        else:
            ax_global[frame+j].bar([0+col,2+col,4+col], means, yerr=std, color=palette[col], alpha=0.5)
    
    j += 1

ax_global[0].set_ylabel('Amplitude (pA)')
ax_global[1].set_ylabel('PPR')
ax_global[2].set_ylabel('Tau (ms)')