# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:57:52 2021

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from neo import io
import seaborn as sns
import scipy.stats as stats

Path = r'E:\AAVDJ.GluSnFR-S72A\iGluSnFRvsEphy\Saturation and desensitization\2021_11_23'

files = sorted(os.listdir(Path))

df_traces = pd.DataFrame(index=None)

LIST_FILES = []
AMP1, AMP2, AMP3 = [],[],[]

for file in range(len(files)):
    if '.wcp' in files[file]:
        LIST_FILES.append(files[file])
        reader = io.WinWcpIO('{}/{}'.format(Path,files[file])) 
        block = reader.read_block()
        
        REC = []
        for episode in block.segments:
            rec = episode.analogsignals[0].magnitude
            time = episode.analogsignals[0].times
            
            LeakStart = np.ravel(np.where(time >= 0.05))[0]
            LeakStop = np.ravel(np.where(time <= 0.15))[-1]
            Leak = np.mean(rec[LeakStart:LeakStop],axis=0)
            RecWithoutLeak = rec-Leak
            REC.append(RecWithoutLeak.reshape(-1,))
        
        average = np.mean(REC, axis=0)
        df_traces[f'{file}'] = average
    
    elif 'Amp.xlsx' in files[file]:
        amps = pd.read_excel('{}/{}'.format(Path,files[file]))
        amp1 = amps.iloc[:,1].values
        amp2 = amps.iloc[:,2].values
        amp3 = amps.iloc[:,3].values
        AMP1.append(abs(amp1))
        AMP2.append(abs(amp2))
        AMP3.append(abs(amp3))
        

name_fig_traces = LIST_FILES[0].rsplit('.',1)[0].rsplit('_',1)[0]
plt.figure()
[plt.plot(time[3509:8187], df_traces.iloc[:,i][3509:8187]) for i in range(len(df_traces.columns))]
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (pA)')
plt.savefig(f'{Path}\{name_fig_traces}_traces.pdf')

LIST_AMPS = [AMP1, AMP2, AMP3]
x_labels = ['1.5mM', '4mM', '4mM + kyn (100µM)', '4mM + kyn (200µM)', '4mM + kyn (300µM)', '4mM + kyn (500µM)']
for item in range(len(LIST_AMPS)):
    fig, ax = plt.subplots(tight_layout=True)
    sns.boxplot(data=LIST_AMPS[item], ax=ax)
    ax.set_title(f'AMP{item+1}')
    ax.set_ylabel('Amplitude (pA)')
    ax.set_xticklabels(x_labels, rotation=40)
    ax.xaxis.set_tick_params(labelsize=7)
    name = LIST_FILES[item].rsplit('.',1)[0]
    
    plt.savefig(f'{Path}\{name_fig_traces}_AMP{item+1}.pdf')
    


LIST_PPR = [AMP2[i]/AMP1[i] for i in range(len(AMP2))]
fig_ppr, ax_ppr = plt.subplots(tight_layout=True)
sns.boxplot(data=LIST_PPR, ax=ax_ppr)
ax_ppr.set_title('PPR2/1')
ax_ppr.set_ylabel('Amplitude (pA)')
ax_ppr.set_xticklabels(x_labels, rotation=40)
ax_ppr.xaxis.set_tick_params(labelsize=7)
plt.savefig(f'{Path}\{name_fig_traces}_PPR.pdf')