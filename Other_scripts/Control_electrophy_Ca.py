# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:57:05 2020

@author: Theo.ROSSI
"""


import numpy as np
import os
from neo import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
from scipy import stats



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

Path_traces = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Controls\Controle_electrophy_Ca\Electrophy_traces'
Path_amp = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Controls\Controle_electrophy_Ca\Electrophy_AMPs+TAUs'
SingleCell = 'PC2'
Cell = 'PC'
Freq = '50Hz'
Ca25 = False
save = False

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def perm_t_test(set1, set2, permutation=10000, histogram=True):
    
    '''
    Set1: first dataset
    Set2: second dataset
    Permutation: number of iterations
    Histogram: histogram of distribution after the permutation
    
    return: p-value
    
    /!\ Set1 and Set2 must be equal size
    '''
    
    
    diff = np.mean(set2) - np.mean(set1)
    print('Original mean: {}'.format(diff))
    
    
    DIFF_ALL, DIFF_UP, DIFF_DOWN = [],[],[]
    np.random.seed(1)
    
    for i in range(permutation):
        
        combination = np.concatenate((set1, set2), axis=None)
        shuffle = np.random.shuffle(combination)
        
        new_set1 = combination[0:int(len(combination)/2)]
        new_set2 = combination[int(len(combination)/2):]
        
        new_diff = np.mean(new_set2) - np.mean(new_set1)
        DIFF_ALL.append(new_diff)
        
        if new_diff >= diff:
            DIFF_UP.append(new_diff)
            
        elif new_diff <= -diff:
            DIFF_DOWN.append(new_diff)    
        
    p_value = (np.sum(DIFF_UP) + np.sum(DIFF_DOWN))/permutation
    
    if histogram == True:
        
        plt.figure()
        plt.hist(DIFF_ALL, bins=int(np.sqrt(permutation)))
        
        if abs(p_value) <= 0.05:
            plt.axvline(diff, color='r', linestyle='--')
        elif abs(p_value) > 0.05:
            plt.axvline(diff, color='k', linestyle='--')
        
        plt.title('p_value: {:.3f}'.format(p_value))
        
    return p_value



files_traces = sorted(os.listdir(Path_traces))
files_amp = sorted(os.listdir(Path_amp))

FILES = []
REC15, REC25, REC4 = [],[],[]

for file in range(len(files_traces)):
    if SingleCell in files_traces[file]:
        if Freq in files_traces[file]:
            FILES.append(files_traces[file])
            reader = io.WinWcpIO('{}/{}'.format(Path_traces,files_traces[file])) 
            block = reader.read_block()
            
            for episode in block.segments:
                rec = episode.analogsignals[0].magnitude
                time = episode.analogsignals[0].times
                
                LeakStart = np.ravel(np.where(time <= 0.1))[-1]
                LeakStop = np.ravel(np.where(time <= 0.15))[-1]
                Leak = np.mean(rec[LeakStart:LeakStop],axis=0)
                RecWithoutLeak = rec-Leak
            
                if '1.5mM' in files_traces[file]:
                    REC15.append(RecWithoutLeak)
                
                if Ca25 == True:
                    if '2.5mM' in files_traces[file]:
                        REC25.append(RecWithoutLeak)
                        
                if '4mM' in files_traces[file]:
                    REC4.append(RecWithoutLeak)


fig, ax = plt.subplots(3,3, figsize=(10,7), tight_layout=True)
fig_averages, ax_averages = plt.subplots(figsize=(10,7), tight_layout=True)

xlabels = ['A1','A2','A3']
Colors = ['deepskyblue','limegreen','purple']

if Ca25 == True:
    [ax[1,0].plot(time, REC25[trace25], 'k', alpha=0.4) for trace25 in range(len(REC25))]
    avg25 = np.average(REC25,axis=0)
    ax[1,0].plot(time, avg25, c=Colors[1])
    ax_averages.plot(time,avg25,c=Colors[1])


[ax[0,0].plot(time, REC15[trace15], 'k', alpha=0.4) for trace15 in range(len(REC15))]
avg15 = np.average(REC15,axis=0)
ax[0,0].plot(time, avg15, c=Colors[0])
ax_averages.plot(time,avg15, c=Colors[0])
        
[ax[2,0].plot(time, REC4[trace4], 'k', alpha=0.4) for trace4 in range(len(REC4))]
avg4 = np.average(REC4,axis=0)
ax[2,0].plot(time, avg4, c=Colors[2])
ax_averages.plot(time,avg4, c=Colors[2])

ax[0,0].set_title('{} traces'.format(SingleCell))
ax[0,1].set_title('{} amps'.format(SingleCell))
ax[0,0].set_ylabel('Amplitude (pA)')

   

AMP1_15, AMP2_15, AMP3_15 = [],[],[]
AMP1_25, AMP2_25, AMP3_25 = [],[],[]
AMP1_4, AMP2_4, AMP3_4 = [],[],[]
MEAN_AMP1_15, MEAN_AMP2_15, MEAN_AMP3_15 = [],[],[]
MEAN_AMP1_25, MEAN_AMP2_25, MEAN_AMP3_25 = [],[],[]
MEAN_AMP1_4, MEAN_AMP2_4, MEAN_AMP3_4 = [],[],[]
MEAN_PPR21_15, MEAN_PPR21_25, MEAN_PPR21_4 = [],[],[]
MEAN_PPR32_15, MEAN_PPR32_25, MEAN_PPR32_4 = [],[],[]
   
for file in range(len(files_amp)):
    if 'Amp' in files_amp[file]:
        if Cell in files_amp[file]:
            if Freq in files_amp[file]:
                data = pd.read_excel('{}/{}'.format(Path_amp,files_amp[file]), index_col=None)
                
                amp1 = abs(data.iloc[1:,1].values)
                amp2 = abs(data.iloc[1:,2].values)
                amp3 = abs(data.iloc[1:,3].values)
                
                mean1 = np.mean(amp1)
                mean2 = np.mean(amp2)
                mean3 = np.mean(amp3)
                
                PPR21 = mean2/mean1
                PPR32 = mean3/mean2
                
                mean_ppr21 = np.mean(PPR21)
                mean_ppr32 = np.mean(PPR32)
                
                Df =  pd.concat((pd.DataFrame(amp1), pd.DataFrame(amp2), pd.DataFrame(amp3)),axis=1)
                Df.columns = ['A1','A2','A3']
                
                if '1.5mM' in files_amp[file]:
                    if SingleCell in files_amp[file]:
                        
                        AMP1_15.append(amp1)
                        AMP2_15.append(amp2)
                        AMP3_15.append(amp3)
                        
                        sns.boxplot(data=Df, showmeans=True, ax=ax[0,1], palette=[Colors[0]], meanprops={'markerfacecolor':'red',
                                                                                                              'markeredgecolor':'black',
                                                                                                              'markersize':'8'})
                        add_stat_annotation(ax[0,1], data=Df, box_pairs=[('A1','A2'), ('A1','A3'), ('A2','A3')],
                                            test='Wilcoxon', text_format='star', verbose=2)
                        
                        for amp in range(amp1.shape[0]):
                            ax[0,1].scatter([xlabels[0],xlabels[1],xlabels[2]],[amp1[amp],amp2[amp],amp3[amp]], color=Colors[0], edgecolors='k', s=10, alpha=0.6)
                            ax[0,1].plot([xlabels[0],xlabels[1],xlabels[2]],[amp1[amp],amp2[amp],amp3[amp]], color='k', alpha=0.2)
                            ax[0,2].scatter(xlabels[0],amp1[amp], color=Colors[0], edgecolors='k', s=10, alpha=0.6)
                            ax[1,2].scatter(xlabels[0],amp2[amp], color=Colors[0], edgecolors='k', s=10, alpha=0.6)
                            ax[2,2].scatter(xlabels[0],amp3[amp], color=Colors[0], edgecolors='k', s=10, alpha=0.6)
                            
                            
                    MEAN_AMP1_15.append(mean1)
                    MEAN_AMP2_15.append(mean2)
                    MEAN_AMP3_15.append(mean3)
                    MEAN_PPR21_15.append(mean_ppr21)
                    MEAN_PPR32_15.append(mean_ppr32)
                    
                    
                if Ca25 == True:    
                    if '2.5mM' in files_amp[file]:
                        if SingleCell in files_amp[file]:
                            
                            AMP1_25.append(amp1)
                            AMP2_25.append(amp2)
                            AMP3_25.append(amp3)
                            
                            sns.boxplot(data=Df, showmeans=True, ax=ax[1,1], palette=[Colors[1]], meanprops={'markerfacecolor':'red',
                                                                                                                    'markeredgecolor':'black',
                                                                                                                    'markersize':'8'})
                            add_stat_annotation(ax[1,1], data=Df, box_pairs=[('A1','A2'), ('A1','A3'), ('A2','A3')],
                                                test='Wilcoxon', text_format='star', verbose=2)
                            
                            for amp in range(amp1.shape[0]):
                               ax[1,1].scatter([xlabels[0],xlabels[1],xlabels[2]],[amp1[amp],amp2[amp],amp3[amp]], color=Colors[1], edgecolors='k', s=10, alpha=0.6)
                               ax[1,1].plot([xlabels[0],xlabels[1],xlabels[2]],[amp1[amp],amp2[amp],amp3[amp]], color='k', alpha=0.2)
                               ax[0,2].scatter(xlabels[1],amp1[amp], color=Colors[1], edgecolors='k', s=10, alpha=0.6)
                               ax[1,2].scatter(xlabels[1],amp2[amp], color=Colors[1], edgecolors='k', s=10, alpha=0.6)
                               ax[2,2].scatter(xlabels[1],amp3[amp], color=Colors[1], edgecolors='k', s=10, alpha=0.6)
                               
                               
                        MEAN_AMP1_25.append(mean1)
                        MEAN_AMP2_25.append(mean2)
                        MEAN_AMP3_25.append(mean3)
                        MEAN_PPR21_25.append(mean_ppr21)
                        MEAN_PPR32_25.append(mean_ppr32)
                    
                    
                if '4mM' in files_amp[file]:
                    
                    plt.figure()
                    plt.plot(np.arange(1,len(data['AMP1'])+1), data['AMP1'], 'b', label='AMP1')
                    plt.plot(np.arange(1,len(data['AMP2'])+1), data['AMP2'], 'r', label='AMP2')
                    plt.plot(np.arange(1,len(data['AMP3'])+1), data['AMP3'], 'g', label='AMP3')
                    plt.title(f'{files_amp[file]}')
                    plt.legend()
                
                    if SingleCell in files_amp[file]:
                        
                        AMP1_4.append(amp1)
                        AMP2_4.append(amp2)
                        AMP3_4.append(amp3)
                        
                        sns.boxplot(data=Df, showmeans=True, ax=ax[2,1], palette=[Colors[2]], meanprops={'markerfacecolor':'red',
                                                                                                            'markeredgecolor':'black',
                                                                                                            'markersize':'8'})
                        add_stat_annotation(ax[2,1], data=Df, box_pairs=[('A1','A2'), ('A1','A3'), ('A2','A3')],
                                            test='Wilcoxon', text_format='star', verbose=2)

                        for amp in range(amp1.shape[0]):
                           ax[2,1].scatter([xlabels[0],xlabels[1],xlabels[2]],[amp1[amp],amp2[amp],amp3[amp]], color=Colors[2], edgecolors='k', s=10, alpha=0.6)
                           ax[2,1].plot([xlabels[0],xlabels[1],xlabels[2]],[amp1[amp],amp2[amp],amp3[amp]], color='k', alpha=0.2)
                           ax[0,2].scatter(xlabels[2],amp1[amp], color=Colors[2], edgecolors='k', s=10, alpha=0.6)
                           ax[1,2].scatter(xlabels[2],amp2[amp], color=Colors[2], edgecolors='k', s=10, alpha=0.6)
                           ax[2,2].scatter(xlabels[2],amp3[amp], color=Colors[2], edgecolors='k', s=10, alpha=0.6)
                           
                           
                    MEAN_AMP1_4.append(mean1)
                    MEAN_AMP2_4.append(mean2)
                    MEAN_AMP3_4.append(mean3)
                    MEAN_PPR21_4.append(mean_ppr21)
                    MEAN_PPR32_4.append(mean_ppr32)
                
                
                
fig1, ax1 = plt.subplots(figsize=(5,6), tight_layout=True)
fig2, ax2 = plt.subplots(1,2,figsize=(5,6), tight_layout=True, sharey=True)
        
if Ca25 == True:
    labels = ['A1-1.5','A1-2.5','A1-4','A2-1.5','A2-2.5','A2-4','A3-1.5','A3-2.5','A3-4',
              'PPR2/1-1.5','PPR2/1-2.5','PPR2/1-4',
              'PPR3/2-1.5','PPR3/2-2.5','PPR3/2-4']
    Palette = ['deepskyblue','limegreen','purple','deepskyblue','limegreen','purple','deepskyblue','limegreen','purple']
    
    AmpDf = pd.concat((pd.DataFrame(AMP1_15[0]),pd.DataFrame(AMP1_25[0]),pd.DataFrame(AMP1_4[0]),
                        pd.DataFrame(AMP2_15[0]),pd.DataFrame(AMP2_25[0]),pd.DataFrame(AMP2_4[0]),
                        pd.DataFrame(AMP3_15[0]),pd.DataFrame(AMP3_25[0]),pd.DataFrame(AMP3_4[0])),axis=1)
    AmpDf.columns = labels[:9]
    
    sns.boxplot(data=[AmpDf.iloc[:,0],AmpDf.iloc[:,1],AmpDf.iloc[:,2]],
                showmeans=True, ax=ax[0,2], palette=[Palette[0],Palette[1],Palette[2]], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})

    sns.boxplot(data=[AmpDf.iloc[:,3],AmpDf.iloc[:,4],AmpDf.iloc[:,5]],
                showmeans=True, ax=ax[1,2], palette=[Palette[0],Palette[1],Palette[2]], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})

    sns.boxplot(data=[AmpDf.iloc[:,6],AmpDf.iloc[:,7],AmpDf.iloc[:,8]],
                showmeans=True, ax=ax[2,2], palette=[Palette[0],Palette[1],Palette[2]], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})
    
    
    MeansDf = pd.concat((pd.DataFrame(MEAN_AMP1_15), pd.DataFrame(MEAN_AMP1_25), pd.DataFrame(MEAN_AMP1_4),
                   pd.DataFrame(MEAN_AMP2_15), pd.DataFrame(MEAN_AMP2_25), pd.DataFrame(MEAN_AMP2_4),
                   pd.DataFrame(MEAN_AMP3_15), pd.DataFrame(MEAN_AMP3_25), pd.DataFrame(MEAN_AMP3_4),
                   pd.DataFrame(MEAN_PPR21_15), pd.DataFrame(MEAN_PPR21_25), pd.DataFrame(MEAN_PPR21_4),
                   pd.DataFrame(MEAN_PPR32_15), pd.DataFrame(MEAN_PPR32_25), pd.DataFrame(MEAN_PPR32_4)), axis=1)
    MeansDf.columns = labels
    
    sns.boxplot(data=[MeansDf.iloc[:,0],MeansDf.iloc[:,1],MeansDf.iloc[:,2],MeansDf.iloc[:,3],MeansDf.iloc[:,4],MeansDf.iloc[:,5],MeansDf.iloc[:,6],MeansDf.iloc[:,7],MeansDf.iloc[:,8]],
                showmeans=True, ax=ax1, palette=Palette, meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})
    
    sns.boxplot(data=[MeansDf.iloc[:,9],MeansDf.iloc[:,10],MeansDf.iloc[:,11]],
                showmeans=True, ax=ax2[0], palette=[Palette[0],Palette[1],Palette[2]], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})
    
    sns.boxplot(data=[MeansDf.iloc[:,12],MeansDf.iloc[:,13],MeansDf.iloc[:,14]],
                showmeans=True, ax=ax2[1], palette=[Palette[0],Palette[1],Palette[2]], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})
    
    
    print('AMPLITUDES COMPARISON')
    print('------------------')
    add_stat_annotation(ax1, data=MeansDf, box_pairs=[('A1-1.5','A1-2.5')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax1, data=MeansDf, box_pairs=[('A1-1.5','A1-4')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax1, data=MeansDf, box_pairs=[('A1-2.5','A1-4')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax1, data=MeansDf, box_pairs=[('A2-1.5','A2-2.5')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax1, data=MeansDf, box_pairs=[('A2-1.5','A2-4')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax1, data=MeansDf, box_pairs=[('A2-2.5','A2-4')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax1, data=MeansDf, box_pairs=[('A3-1.5','A3-2.5')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax1, data=MeansDf, box_pairs=[('A3-1.5','A3-4')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax1, data=MeansDf, box_pairs=[('A3-2.5','A3-4')], test='Wilcoxon', text_format='full', verbose=2)
    print('------------------')
    
    print('PPR comparison')
    print('------------------')
    add_stat_annotation(ax2[0], data=MeansDf.iloc[:,9:12], box_pairs=[('PPR2/1-1.5','PPR2/1-2.5')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax2[0], data=MeansDf.iloc[:,9:12], box_pairs=[('PPR2/1-1.5','PPR2/1-4')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax2[0], data=MeansDf.iloc[:,9:12], box_pairs=[('PPR2/1-2.5','PPR2/1-4')], test='Wilcoxon', text_format='full', verbose=2)
    
    add_stat_annotation(ax2[1], data=MeansDf.iloc[:,12:], box_pairs=[('PPR3/2-1.5','PPR3/2-2.5')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax2[1], data=MeansDf.iloc[:,12:], box_pairs=[('PPR3/2-1.5','PPR3/2-4')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax2[1], data=MeansDf.iloc[:,12:], box_pairs=[('PPR3/2-2.5','PPR3/2-4')], test='Wilcoxon', text_format='full', verbose=2)
    print('------------------')
    
    
    for i in range(MeansDf.shape[0]):
        data1 = MeansDf.iloc[i,:6].values
        data2 = MeansDf.iloc[i,6:8].values
        data3 = MeansDf.iloc[i,8:].values
        
        ax1.scatter([0,1,2],[data1[0],data1[1],data1[2]], color=[Palette[0],Palette[1],Palette[2]], edgecolors='k', s=20, alpha=0.6)
        ax1.scatter([3,4,5],[data1[3],data1[4],data1[5]], color=[Palette[0],Palette[1],Palette[2]], edgecolors='k', s=20, alpha=0.6)
        ax1.scatter([6,7,8],[data1[6],data1[7],data1[8]], color=[Palette[0],Palette[1],Palette[2]], edgecolors='k', s=20, alpha=0.6)
        
        ax1.plot([0,1,2],[data1[0],data1[1],data1[2]], color='k', alpha=0.4)
        ax1.plot([3,4,5],[data1[3],data1[4],data1[5]], color='k', alpha=0.4)
        ax1.plot([6,7,8],[data1[6],data1[7],data1[8]], color='k', alpha=0.4)
        
        ax2[0].scatter([0,1,2],[data2[0],data2[1],data2[2]], color=[Palette[0],Palette[1],Palette[2]], edgecolors='k', s=20, alpha=0.6)
        ax2[1].scatter([0,1,2],[data3[0],data3[1],data3[2]], color=[Palette[0],Palette[1],Palette[2]], edgecolors='k', s=20, alpha=0.6)
        ax2[0].plot([0,1,2],[data2[0],data2[1],data2[2]], color='k', alpha=0.4)
        ax2[1].plot([0,1,2],[data3[0],data3[1],data3[2]], color='k', alpha=0.4)
        
    ax1.set_title('Mean amp for {}s at {}'.format(Cell,Freq)), ax2[0].set_title('Mean PPR2/1\n{}s at {}'.format(Cell,Freq)), ax2[1].set_title('Mean PPR3/2\n{}s at {}'.format(Cell,Freq))
    ax1.set_ylabel('Amp (pA)'), ax2[0].set_ylabel('A2/A1'), ax2[1].set_ylabel('A3/A2')
    



else:
    labels = ['A1-1.5','A1-4','A2-1.5','A2-4','A3-1.5','A3-4',
              'PPR2/1-1.5','PPR2/1-4',
              'PPR3/2-1.5','PPR3/2-4']
    Palette = ['deepskyblue','purple','deepskyblue','purple','deepskyblue','purple']
    
    AmpDf = pd.concat((pd.DataFrame(AMP1_15[0]), pd.DataFrame(AMP1_4[0]),
                        pd.DataFrame(AMP2_15[0]),pd.DataFrame(AMP2_4[0]),
                        pd.DataFrame(AMP3_15[0]),pd.DataFrame(AMP3_4[0])),axis=1)
    AmpDf.columns = labels[:6]
    
    sns.boxplot(data=[AmpDf.iloc[:,0], AmpDf.iloc[:,1]],
                showmeans=True, ax=ax[0,2], palette=[Palette[0],Palette[1]], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})
    
    sns.boxplot(data=[AmpDf.iloc[:,2], AmpDf.iloc[:,3]],
                showmeans=True, ax=ax[1,2], palette=[Palette[0],Palette[1]], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})
    
    sns.boxplot(data=[AmpDf.iloc[:,4], AmpDf.iloc[:,5]],
                showmeans=True, ax=ax[2,2], palette=[Palette[0],Palette[1]], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})
    
    
    MeansDf = pd.concat((pd.DataFrame(MEAN_AMP1_15), pd.DataFrame(MEAN_AMP1_4),
                         pd.DataFrame(MEAN_AMP2_15), pd.DataFrame(MEAN_AMP2_4),
                         pd.DataFrame(MEAN_AMP3_15), pd.DataFrame(MEAN_AMP3_4),
                         pd.DataFrame(MEAN_PPR21_15), pd.DataFrame(MEAN_PPR21_4),
                         pd.DataFrame(MEAN_PPR32_15), pd.DataFrame(MEAN_PPR32_4)), axis=1)
    MeansDf.columns = labels
    
    sns.boxplot(data=[MeansDf.iloc[:,0],MeansDf.iloc[:,1],MeansDf.iloc[:,2],MeansDf.iloc[:,3],MeansDf.iloc[:,4],MeansDf.iloc[:,5]],
                showmeans=True, ax=ax1, palette=Palette, meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})
    
    sns.boxplot(data=[MeansDf.iloc[:,6],MeansDf.iloc[:,7]],
                showmeans=True, ax=ax2[0], palette=[Palette[0],Palette[1]], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})
    
    sns.boxplot(data=[MeansDf.iloc[:,8],MeansDf.iloc[:,9]],
                showmeans=True, ax=ax2[1], palette=[Palette[0],Palette[1]], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})
    
    
    print('------------------')
    print('------------------')
    print('AMPLITUDES COMPARISON')
    print('------------------')
    add_stat_annotation(ax1, data=MeansDf, box_pairs=[('A1-1.5','A1-4')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax1, data=MeansDf, box_pairs=[('A2-1.5','A2-4')], test='Wilcoxon', text_format='full', verbose=2)
    add_stat_annotation(ax1, data=MeansDf, box_pairs=[('A3-1.5','A3-4')], test='Wilcoxon', text_format='full', verbose=2)
    print('------------------')
    print('------------------')
    
    print('PPR COMPARISON')
    print('------------------')
    add_stat_annotation(ax2[0], data=MeansDf.iloc[:,6:8], box_pairs=[('PPR2/1-1.5','PPR2/1-4')], test='t-test_paired', text_format='full', verbose=2)
    
    add_stat_annotation(ax2[1], data=MeansDf.iloc[:,8:], box_pairs=[('PPR3/2-1.5','PPR3/2-4')], test='t-test_paired', text_format='full', verbose=2)
    print('------------------')
    print('------------------')


    for i in range(MeansDf.shape[0]):
        data1 = MeansDf.iloc[i,:6].values
        data2 = MeansDf.iloc[i,6:8].values
        data3 = MeansDf.iloc[i,8:].values
        
        ax1.scatter([0,1],[data1[0],data1[1]], color=[Palette[0],Palette[1]], edgecolors='k', s=20, alpha=0.6)
        ax1.scatter([2,3],[data1[2],data1[3]], color=[Palette[0],Palette[1]], edgecolors='k', s=20, alpha=0.6)
        ax1.scatter([4,5],[data1[4],data1[5]], color=[Palette[0],Palette[1]], edgecolors='k', s=20, alpha=0.6)
        
        ax1.plot([0,1],[data1[0],data1[1]], color='k', alpha=0.4)
        ax1.plot([2,3],[data1[2],data1[3]], color='k', alpha=0.4)
        ax1.plot([4,5],[data1[4],data1[5]], color='k', alpha=0.4)
        
        ax2[0].scatter([0,1],[data2[0],data2[1]], color=[Palette[0],Palette[1]], edgecolors='k', s=20, alpha=0.6)
        ax2[1].scatter([0,1],[data3[0],data3[1]], color=[Palette[0],Palette[1]], edgecolors='k', s=20, alpha=0.6)
        ax2[0].plot([0,1],[data2[0],data2[1]], color='k', alpha=0.4)
        ax2[1].plot([0,1],[data3[0],data3[1]], color='k', alpha=0.4)
        
    ax1.set_title('Mean amp for {}s at {}'.format(Cell,Freq))
    ax2[0].set_title('Mean PPR2/1\n{}s at {}'.format(Cell,Freq))
    ax2[1].set_title('Mean PPR3/2\n{}s at {}'.format(Cell,Freq))
    ax1.set_ylabel('Amp (pA)')
    ax2[0].set_ylabel('A2/A1')
    ax2[1].set_ylabel('A3/A2')

    
    
    ### FOLD INCREASE A1    
    fold = MeansDf['A1-4']/MeansDf['A1-1.5']
    fig_fold, ax_fold = plt.subplots()
    sns.histplot(data=fold, ax=ax_fold, binwidth=0.4, stat='probability')
    ax_fold.set_xlim(0)
    ax_fold.set_ylim(0)


if save == True:
    with pd.ExcelWriter('E:\AAVDJ.GluSnFR-S72A\Controls\Controle_electrophy_Ca\Parameters_1.5vs4_{}.xlsx'.format(Freq)) as writer:
        MeansDf.to_excel(writer)
        writer.save() 